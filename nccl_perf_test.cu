// nccl_perf_test.cu
// Standalone NCCL performance test: cudaMemcpy vs grouped NCCL vs isolated NCCL
// GPUs 4-7, 1 GB transfers, 10 iterations each

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <map>
#include <utility>

#define CUDA_CHECK(cmd) do {                                          \
  cudaError_t e = (cmd);                                              \
  if (e != cudaSuccess) {                                             \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
            __FILE__, __LINE__, cudaGetErrorString(e));               \
    exit(EXIT_FAILURE);                                               \
  }                                                                   \
} while(0)

#define NCCL_CHECK(cmd) do {                                          \
  ncclResult_t r = (cmd);                                             \
  if (r != ncclSuccess) {                                             \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                        \
            __FILE__, __LINE__, ncclGetErrorString(r));               \
    exit(EXIT_FAILURE);                                               \
  }                                                                   \
} while(0)

// ── Configuration ──────────────────────────────────────────────────
static const int    GPU_IDS[]   = {4, 5, 6, 7};
static const int    NUM_GPUS    = 4;
static const size_t TRANSFER_SZ     = 1ULL << 30;   // 1 GB  (P2P per-pair size)
// Collective sizes scaled so total bus data = 12 GB in all cases.
// AllReduce bus_data = 2*(N-1)*sz  → sz = 12/(2*3) = 2 GB
// Broadcast bus_data = (N-1)*sz   → sz = 12/3     = 4 GB
// AllGather bus_data = N*(N-1)*sz → sz = 12/12    = 1 GB  (= TRANSFER_SZ)
static const size_t ALLREDUCE_SZ    = 2ULL << 30;   // 2 GB per GPU
static const size_t BROADCAST_SZ    = 4ULL << 30;   // 4 GB per GPU
static const size_t ALLGATHER_SZ    = 1ULL << 30;   // 1 GB per GPU (4 GB recv buffer)
static const int    NUM_ITERS   = 10;

// A transfer pair: src GPU → dst GPU
struct TransferPair {
  int src;   // GPU id
  int dst;   // GPU id
};

// Build a set of transfers that are spread evenly among GPUs 4-7.
// Each GPU sends to every other GPU exactly once → 12 pairs total.
// Each GPU participates in 3 sends + 3 recvs = 6 ops.
static std::vector<TransferPair> buildTransferPairs() {
  std::vector<TransferPair> pairs;
  for (int i = 0; i < NUM_GPUS; ++i)
    for (int j = 0; j < NUM_GPUS; ++j)
      if (i != j)
        pairs.push_back({GPU_IDS[i], GPU_IDS[j]});
  return pairs;
}

// ── Helper: allocate device buffers ────────────────────────────────
struct GpuBuffer {
  void*  ptr;
  int    device;
  size_t size;
};

static GpuBuffer allocBuf(int device, size_t sz) {
  CUDA_CHECK(cudaSetDevice(device));
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, sz));
  CUDA_CHECK(cudaMemset(p, 0x42, sz));  // fill with pattern
  return {p, device, sz};
}

static void freeBuf(GpuBuffer& b) {
  CUDA_CHECK(cudaSetDevice(b.device));
  CUDA_CHECK(cudaFree(b.ptr));
  b.ptr = nullptr;
}

// ── 1. cudaMemcpyPeerAsync benchmark ──────────────────────────────
static double benchCudaMemcpy(const std::vector<TransferPair>& pairs,
                              size_t sz, int iter) {
  // Allocate src + dst buffers for every pair
  std::vector<GpuBuffer> srcs(pairs.size()), dsts(pairs.size());
  std::vector<cudaStream_t> streams(pairs.size());

  for (size_t i = 0; i < pairs.size(); ++i) {
    srcs[i] = allocBuf(pairs[i].src, sz);
    dsts[i] = allocBuf(pairs[i].dst, sz);
    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  // Enable peer access where possible
  for (int i = 0; i < NUM_GPUS; ++i)
    for (int j = 0; j < NUM_GPUS; ++j)
      if (i != j) {
        CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
        cudaDeviceEnablePeerAccess(GPU_IDS[j], 0); // ignore error if already enabled
      }

  // Warm-up / actual iteration
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < pairs.size(); ++i) {
    CUDA_CHECK(cudaMemcpyPeerAsync(dsts[i].ptr, pairs[i].dst,
                                   srcs[i].ptr, pairs[i].src,
                                   sz, streams[i]));
  }
  for (size_t i = 0; i < pairs.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Cleanup
  for (size_t i = 0; i < pairs.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    freeBuf(srcs[i]);
    freeBuf(dsts[i]);
  }
  return ms;
}

// ── 2. NCCL Grouped Send/Recv (single N-rank communicator) ────────
//
// Creates one 4-rank communicator (rank i ↔ GPU_IDS[i]).
// All send/recv calls are wrapped in a single ncclGroupStart/End so
// NCCL can optimise routing across all transfers simultaneously.

using PairKey = std::pair<int,int>;

// Map GPU device id → rank index (0-3)
static std::map<int, int> gpuToRank;
static ncclComm_t groupComms[NUM_GPUS];  // one comm per rank
static bool       groupCommsCreated = false;

static void ensureGroupComms() {
  if (groupCommsCreated) return;

  for (int i = 0; i < NUM_GPUS; ++i)
    gpuToRank[GPU_IDS[i]] = i;

  ncclUniqueId uid;
  NCCL_CHECK(ncclGetUniqueId(&uid));
  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    NCCL_CHECK(ncclCommInitRank(&groupComms[i], NUM_GPUS, uid, i));
  }
  NCCL_CHECK(ncclGroupEnd());
  groupCommsCreated = true;
}

static void destroyGroupComms() {
  if (!groupCommsCreated) return;
  for (int i = 0; i < NUM_GPUS; ++i)
    ncclCommDestroy(groupComms[i]);
  groupCommsCreated = false;
}

static double benchNcclGrouped(const std::vector<TransferPair>& pairs,
                               size_t sz, int iter) {
  std::vector<GpuBuffer> srcs(pairs.size()), dsts(pairs.size());
  // One stream per GPU (shared across all sends/recvs on that GPU)
  std::map<int, cudaStream_t> streams;

  for (int g : GPU_IDS) {
    CUDA_CHECK(cudaSetDevice(g));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    streams[g] = s;
  }

  for (size_t i = 0; i < pairs.size(); ++i) {
    srcs[i] = allocBuf(pairs[i].src, sz);
    dsts[i] = allocBuf(pairs[i].dst, sz);
  }

  ensureGroupComms();

  auto t0 = std::chrono::high_resolution_clock::now();

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < pairs.size(); ++i) {
    int srcRank = gpuToRank[pairs[i].src];
    int dstRank = gpuToRank[pairs[i].dst];

    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    NCCL_CHECK(ncclSend(srcs[i].ptr, sz, ncclChar, /*peer=*/dstRank,
                        groupComms[srcRank], streams[pairs[i].src]));

    CUDA_CHECK(cudaSetDevice(pairs[i].dst));
    NCCL_CHECK(ncclRecv(dsts[i].ptr, sz, ncclChar, /*peer=*/srcRank,
                        groupComms[dstRank], streams[pairs[i].dst]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int g : GPU_IDS) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (size_t i = 0; i < pairs.size(); ++i) {
    freeBuf(srcs[i]);
    freeBuf(dsts[i]);
  }
  for (int g : GPU_IDS) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamDestroy(streams[g]));
  }
  return ms;
}

// ── Per-pair communicator helpers (for isolated + optimized tests) ──

struct PairComm {
  ncclComm_t srcComm; // rank 0
  ncclComm_t dstComm; // rank 1
};

static std::map<PairKey, PairComm>& getCachedPairComms() {
  static std::map<PairKey, PairComm> cache;
  return cache;
}

static PairComm& getOrCreatePairComm(int src, int dst) {
  auto& cache = getCachedPairComms();
  PairKey key{src, dst};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  PairComm pc{};
  ncclUniqueId uid;
  NCCL_CHECK(ncclGetUniqueId(&uid));
  NCCL_CHECK(ncclGroupStart());
  CUDA_CHECK(cudaSetDevice(src));
  NCCL_CHECK(ncclCommInitRank(&pc.srcComm, 2, uid, 0));
  CUDA_CHECK(cudaSetDevice(dst));
  NCCL_CHECK(ncclCommInitRank(&pc.dstComm, 2, uid, 1));
  NCCL_CHECK(ncclGroupEnd());

  cache[key] = pc;
  return cache[key];
}

static void destroyCachedComms() {
  auto& cache = getCachedPairComms();
  for (auto& [key, pc] : cache) {
    ncclCommDestroy(pc.srcComm);
    ncclCommDestroy(pc.dstComm);
  }
  cache.clear();
}

// ── 3. NCCL Isolated Send/Recv (one cached comm per pair, no group) ─
//
// Per-pair 2-rank communicators stored in a static local map so they
// are created exactly once (first call) and reused on every subsequent
// call.  Each pair is issued in its own ncclGroupStart/End from a
// separate thread → concurrent execution.

static std::map<PairKey, PairComm> s_isolatedComms;
static bool s_isolatedCommsCreated = false;

static void ensureIsolatedComms(const std::vector<TransferPair>& pairs) {
  if (s_isolatedCommsCreated) return;
  for (auto& p : pairs) {
    PairKey key{p.src, p.dst};
    if (s_isolatedComms.count(key)) continue;
    PairComm pc{};
    ncclUniqueId uid;
    NCCL_CHECK(ncclGetUniqueId(&uid));
    NCCL_CHECK(ncclGroupStart());
    CUDA_CHECK(cudaSetDevice(p.src));
    NCCL_CHECK(ncclCommInitRank(&pc.srcComm, 2, uid, 0));
    CUDA_CHECK(cudaSetDevice(p.dst));
    NCCL_CHECK(ncclCommInitRank(&pc.dstComm, 2, uid, 1));
    NCCL_CHECK(ncclGroupEnd());
    s_isolatedComms[key] = pc;
  }
  s_isolatedCommsCreated = true;
}

static void destroyIsolatedComms() {
  for (auto& [key, pc] : s_isolatedComms) {
    ncclCommDestroy(pc.srcComm);
    ncclCommDestroy(pc.dstComm);
  }
  s_isolatedComms.clear();
  s_isolatedCommsCreated = false;
}

static double benchNcclIsolated(const std::vector<TransferPair>& pairs,
                                size_t sz, int iter) {
  std::vector<GpuBuffer> srcs(pairs.size()), dsts(pairs.size());
  std::vector<cudaStream_t> sendStreams(pairs.size()), recvStreams(pairs.size());

  for (size_t i = 0; i < pairs.size(); ++i) {
    srcs[i] = allocBuf(pairs[i].src, sz);
    dsts[i] = allocBuf(pairs[i].dst, sz);
    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    CUDA_CHECK(cudaStreamCreate(&sendStreams[i]));
    CUDA_CHECK(cudaSetDevice(pairs[i].dst));
    CUDA_CHECK(cudaStreamCreate(&recvStreams[i]));
  }

  // Create comms once; reuse on every subsequent call
  ensureIsolatedComms(pairs);

  auto t0 = std::chrono::high_resolution_clock::now();

  // Each pair in its own thread with its own ncclGroupStart/End
  std::vector<std::thread> threads;
  for (size_t i = 0; i < pairs.size(); ++i) {
    threads.emplace_back([&, i]() {
      PairKey key{pairs[i].src, pairs[i].dst};
      auto& pc = s_isolatedComms[key];
      NCCL_CHECK(ncclGroupStart());
      CUDA_CHECK(cudaSetDevice(pairs[i].src));
      NCCL_CHECK(ncclSend(srcs[i].ptr, sz, ncclChar, /*peer=*/1,
                          pc.srcComm, sendStreams[i]));
      CUDA_CHECK(cudaSetDevice(pairs[i].dst));
      NCCL_CHECK(ncclRecv(dsts[i].ptr, sz, ncclChar, /*peer=*/0,
                          pc.dstComm, recvStreams[i]));
      NCCL_CHECK(ncclGroupEnd());
    });
  }
  for (auto& t : threads) t.join();

  for (size_t i = 0; i < pairs.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    CUDA_CHECK(cudaStreamSynchronize(sendStreams[i]));
    CUDA_CHECK(cudaSetDevice(pairs[i].dst));
    CUDA_CHECK(cudaStreamSynchronize(recvStreams[i]));
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (size_t i = 0; i < pairs.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    CUDA_CHECK(cudaStreamDestroy(sendStreams[i]));
    CUDA_CHECK(cudaSetDevice(pairs[i].dst));
    CUDA_CHECK(cudaStreamDestroy(recvStreams[i]));
    freeBuf(srcs[i]);
    freeBuf(dsts[i]);
  }
  return ms;
}

// ── 4. NCCL Optimized: N-rank comm + NCCL_P2P_USE_CUDA_MEMCPY ────
//
// Sets NCCL env vars BEFORE creating a fresh 4-rank communicator:
//   NCCL_P2P_USE_CUDA_MEMCPY=1  → forces NCCL to use cudaMemcpy for
//     P2P transfers (bypasses NCCL's own protocol/proxy threads)
//   NCCL_MAX_NCHANNELS=32       → more channels = more parallelism
//   NCCL_MIN_NCHANNELS=32       → don't let NCCL reduce channels
//
// Uses a single grouped call so NCCL can schedule all transfers.

static ncclComm_t optComms[NUM_GPUS];
static bool       optCommsCreated = false;

static void ensureOptComms() {
  if (optCommsCreated) return;

  ncclUniqueId uid;
  NCCL_CHECK(ncclGetUniqueId(&uid));
  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    NCCL_CHECK(ncclCommInitRank(&optComms[i], NUM_GPUS, uid, i));
  }
  NCCL_CHECK(ncclGroupEnd());
  optCommsCreated = true;
}

static void destroyOptComms() {
  if (!optCommsCreated) return;
  for (int i = 0; i < NUM_GPUS; ++i)
    ncclCommDestroy(optComms[i]);
  optCommsCreated = false;
}

static double benchNcclOptimized(const std::vector<TransferPair>& pairs,
                                 size_t sz, int iter) {
  std::vector<GpuBuffer> srcs(pairs.size()), dsts(pairs.size());
  std::map<int, cudaStream_t> streams;

  for (int g : GPU_IDS) {
    CUDA_CHECK(cudaSetDevice(g));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    streams[g] = s;
  }

  for (size_t i = 0; i < pairs.size(); ++i) {
    srcs[i] = allocBuf(pairs[i].src, sz);
    dsts[i] = allocBuf(pairs[i].dst, sz);
  }

  ensureOptComms();

  auto t0 = std::chrono::high_resolution_clock::now();

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < pairs.size(); ++i) {
    int srcRank = gpuToRank[pairs[i].src];
    int dstRank = gpuToRank[pairs[i].dst];

    CUDA_CHECK(cudaSetDevice(pairs[i].src));
    NCCL_CHECK(ncclSend(srcs[i].ptr, sz, ncclChar, /*peer=*/dstRank,
                        optComms[srcRank], streams[pairs[i].src]));

    CUDA_CHECK(cudaSetDevice(pairs[i].dst));
    NCCL_CHECK(ncclRecv(dsts[i].ptr, sz, ncclChar, /*peer=*/srcRank,
                        optComms[dstRank], streams[pairs[i].dst]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int g : GPU_IDS) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (size_t i = 0; i < pairs.size(); ++i) {
    freeBuf(srcs[i]);
    freeBuf(dsts[i]);
  }
  for (int g : GPU_IDS) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamDestroy(streams[g]));
  }
  return ms;
}

// ── Shared collective communicator (4-rank, one per GPU) ──────────
// Reused by AllReduce, Broadcast, and AllGather benchmarks.
// Created lazily, destroyed explicitly.

static ncclComm_t collComms[NUM_GPUS];
static bool       collCommsCreated = false;

static void ensureCollComms() {
  if (collCommsCreated) return;
  ncclUniqueId uid;
  NCCL_CHECK(ncclGetUniqueId(&uid));
  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    NCCL_CHECK(ncclCommInitRank(&collComms[i], NUM_GPUS, uid, i));
  }
  NCCL_CHECK(ncclGroupEnd());
  collCommsCreated = true;
}

static void destroyCollComms() {
  if (!collCommsCreated) return;
  for (int i = 0; i < NUM_GPUS; ++i)
    ncclCommDestroy(collComms[i]);
  collCommsCreated = false;
}

// ── 5. NCCL AllReduce ─────────────────────────────────────────────
// Each GPU contributes TRANSFER_SZ bytes; result summed across all
// GPUs and written back to every GPU's buffer.
// Total data moved per iteration ≈ 2×(N-1)/N × N×sz = 2×(N-1)×sz.

static double benchNcclAllReduce(size_t sz, int iter) {
  // sz here is bytes; treat as float32 elements
  size_t nElems = sz / sizeof(float);
  std::vector<GpuBuffer> bufs(NUM_GPUS);
  std::vector<cudaStream_t> streams(NUM_GPUS);

  for (int i = 0; i < NUM_GPUS; ++i) {
    bufs[i] = allocBuf(GPU_IDS[i], sz);
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  ensureCollComms();

  auto t0 = std::chrono::high_resolution_clock::now();

  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    NCCL_CHECK(ncclAllReduce(bufs[i].ptr, bufs[i].ptr,
                             nElems, ncclFloat, ncclSum,
                             collComms[i], streams[i]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    freeBuf(bufs[i]);
  }
  return ms;
}

// ── 6. NCCL Broadcast ─────────────────────────────────────────────
// Root (rank 0 = GPU_IDS[0]) broadcasts TRANSFER_SZ bytes to all
// other GPUs.

static double benchNcclBroadcast(size_t sz, int iter) {
  size_t nElems = sz / sizeof(float);
  std::vector<GpuBuffer> bufs(NUM_GPUS);
  std::vector<cudaStream_t> streams(NUM_GPUS);

  for (int i = 0; i < NUM_GPUS; ++i) {
    bufs[i] = allocBuf(GPU_IDS[i], sz);
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  ensureCollComms();

  auto t0 = std::chrono::high_resolution_clock::now();

  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    NCCL_CHECK(ncclBroadcast(bufs[i].ptr, bufs[i].ptr,
                             nElems, ncclFloat, /*root=*/0,
                             collComms[i], streams[i]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    freeBuf(bufs[i]);
  }
  return ms;
}

// ── 7. NCCL AllGather ─────────────────────────────────────────────
// Each GPU contributes a TRANSFER_SZ-byte slice; every GPU receives
// all N slices → each output buffer is N×TRANSFER_SZ bytes.

static double benchNcclAllGather(size_t sz, int iter) {
  size_t nElems = sz / sizeof(float);  // per-GPU send count
  std::vector<GpuBuffer> sendBufs(NUM_GPUS), recvBufs(NUM_GPUS);
  std::vector<cudaStream_t> streams(NUM_GPUS);

  for (int i = 0; i < NUM_GPUS; ++i) {
    sendBufs[i] = allocBuf(GPU_IDS[i], sz);
    recvBufs[i] = allocBuf(GPU_IDS[i], sz * NUM_GPUS);
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  ensureCollComms();

  auto t0 = std::chrono::high_resolution_clock::now();

  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    NCCL_CHECK(ncclAllGather(sendBufs[i].ptr, recvBufs[i].ptr,
                             nElems, ncclFloat,
                             collComms[i], streams[i]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDA_CHECK(cudaSetDevice(GPU_IDS[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    freeBuf(sendBufs[i]);
    freeBuf(recvBufs[i]);
  }
  return ms;
}

// ── Main ──────────────────────────────────────────────────────────
int main() {
  // Verify we have enough GPUs
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 8) {
    fprintf(stderr, "Need at least 8 GPUs (using 4-7), but only %d found.\n",
            deviceCount);
    return 1;
  }

  auto pairs = buildTransferPairs();
  printf("=== NCCL Performance Test ===\n");
  printf("GPUs: 4, 5, 6, 7\n");
  printf("Transfer size: 1 GB per pair\n");
  printf("Transfer pairs: %zu (every GPU → every other GPU)\n", pairs.size());
  printf("Iterations: %d\n\n", NUM_ITERS);

  // Storage for timing results
  std::vector<double> cudaMemcpyTimes(NUM_ITERS);
  std::vector<double> ncclGroupedTimes(NUM_ITERS);
  std::vector<double> ncclIsolatedTimes(NUM_ITERS);
  std::vector<double> ncclOptimizedTimes(NUM_ITERS);
  std::vector<double> allReduceTimes(NUM_ITERS);
  std::vector<double> broadcastTimes(NUM_ITERS);
  std::vector<double> allGatherTimes(NUM_ITERS);

  // ── Benchmark 1: cudaMemcpyPeerAsync ──
  printf("Running cudaMemcpyPeerAsync...\n");
  for (int i = 0; i < NUM_ITERS; ++i) {
    cudaMemcpyTimes[i] = benchCudaMemcpy(pairs, TRANSFER_SZ, i);
    printf("  iter %2d: %8.2f ms\n", i, cudaMemcpyTimes[i]);
  }

  // ── Benchmark 2: NCCL Grouped Send/Recv ──
  printf("Running NCCL Grouped Send/Recv...\n");
  for (int i = 0; i < NUM_ITERS; ++i) {
    ncclGroupedTimes[i] = benchNcclGrouped(pairs, TRANSFER_SZ, i);
    printf("  iter %2d: %8.2f ms\n", i, ncclGroupedTimes[i]);
  }

  destroyGroupComms();

  // ── Benchmark 3: NCCL Isolated Send/Recv ──
  printf("Running NCCL Isolated Send/Recv (per-pair 2-rank comms)...\n");
  for (int i = 0; i < NUM_ITERS; ++i) {
    ncclIsolatedTimes[i] = benchNcclIsolated(pairs, TRANSFER_SZ, i);
    printf("  iter %2d: %8.2f ms\n", i, ncclIsolatedTimes[i]);
  }
  destroyIsolatedComms();

  // ── Benchmark 4: NCCL Optimized ──
  // Set env vars BEFORE creating new communicators
  setenv("NCCL_P2P_USE_CUDA_MEMCPY", "1", 1);
  setenv("NCCL_MAX_NCHANNELS",       "32", 1);
  setenv("NCCL_MIN_NCHANNELS",       "32", 1);
  printf("Running NCCL Optimized (P2P_USE_CUDA_MEMCPY=1, NCHANNELS=32)...\n");
  for (int i = 0; i < NUM_ITERS; ++i) {
    ncclOptimizedTimes[i] = benchNcclOptimized(pairs, TRANSFER_SZ, i);
    printf("  iter %2d: %8.2f ms\n", i, ncclOptimizedTimes[i]);
  }
  destroyOptComms();
  // Restore env
  unsetenv("NCCL_P2P_USE_CUDA_MEMCPY");
  unsetenv("NCCL_MAX_NCHANNELS");
  unsetenv("NCCL_MIN_NCHANNELS");

  // ── Benchmark 5: AllReduce ──
  printf("Running NCCL AllReduce (2 GB per GPU → 12 GB bus data, ncclFloat+ncclSum)...\n");
  for (int i = 0; i < NUM_ITERS; ++i) {
    allReduceTimes[i] = benchNcclAllReduce(ALLREDUCE_SZ, i);
    printf("  iter %2d: %8.2f ms\n", i, allReduceTimes[i]);
  }

  // ── Benchmark 6: Broadcast ──
  printf("Running NCCL Broadcast (4 GB from GPU %d → 12 GB bus data)...\n", GPU_IDS[0]);
  for (int i = 0; i < NUM_ITERS; ++i) {
    broadcastTimes[i] = benchNcclBroadcast(BROADCAST_SZ, i);
    printf("  iter %2d: %8.2f ms\n", i, broadcastTimes[i]);
  }

  // ── Benchmark 7: AllGather ──
  printf("Running NCCL AllGather (1 GB per GPU → 4 GB recv, 12 GB bus data)...\n");
  for (int i = 0; i < NUM_ITERS; ++i) {
    allGatherTimes[i] = benchNcclAllGather(ALLGATHER_SZ, i);
    printf("  iter %2d: %8.2f ms\n", i, allGatherTimes[i]);
  }

  destroyCollComms();

  // ── Results ──
  auto avg = [](const std::vector<double>& v, int skip) -> double {
    double s = 0;
    int    n = 0;
    for (int i = skip; i < (int)v.size(); ++i) { s += v[i]; ++n; }
    return n > 0 ? s / n : 0;
  };
  auto minv = [](const std::vector<double>& v, int skip) -> double {
    double m = 1e18;
    for (int i = skip; i < (int)v.size(); ++i) if (v[i] < m) m = v[i];
    return m;
  };
  auto maxv = [](const std::vector<double>& v, int skip) -> double {
    double m = 0;
    for (int i = skip; i < (int)v.size(); ++i) if (v[i] > m) m = v[i];
    return m;
  };

  printf("\n");
  printf("\n");
  printf("================================================================\n");
  printf("  POINT-TO-POINT  (12 pairs x 1 GB = 12 GB per iteration)\n");
  printf("================================================================\n");
  printf("  %-36s  avg(all) %7.2f ms  avg(skip1) %7.2f ms  min %7.2f  max %7.2f\n",
         "cudaMemcpyPeerAsync",
         avg(cudaMemcpyTimes, 0), avg(cudaMemcpyTimes, 1),
         minv(cudaMemcpyTimes, 1), maxv(cudaMemcpyTimes, 1));
  printf("  %-36s  avg(all) %7.2f ms  avg(skip1) %7.2f ms  min %7.2f  max %7.2f\n",
         "NCCL Grouped Send/Recv (4-rank comm)",
         avg(ncclGroupedTimes, 0), avg(ncclGroupedTimes, 1),
         minv(ncclGroupedTimes, 1), maxv(ncclGroupedTimes, 1));
  printf("  %-36s  avg(all) %7.2f ms  avg(skip1) %7.2f ms  min %7.2f  max %7.2f\n",
         "NCCL Isolated Send/Recv (per-pair comms)",
         avg(ncclIsolatedTimes, 0), avg(ncclIsolatedTimes, 1),
         minv(ncclIsolatedTimes, 1), maxv(ncclIsolatedTimes, 1));
  printf("  %-36s  avg(all) %7.2f ms  avg(skip1) %7.2f ms  min %7.2f  max %7.2f\n",
         "NCCL Optimized (P2P_CUDA_MEMCPY+32ch)",
         avg(ncclOptimizedTimes, 0), avg(ncclOptimizedTimes, 1),
         minv(ncclOptimizedTimes, 1), maxv(ncclOptimizedTimes, 1));
  printf("\n");
  printf("================================================================\n");
  printf("  COLLECTIVES  (12 GB bus data each, 4 GPUs)\n");
  printf("================================================================\n");
  double collGB = 12.0;
  printf("  %-36s  data_moved %4.1f GB  avg(all) %7.2f ms  avg(skip1) %7.2f ms  min %7.2f  max %7.2f\n",
         "NCCL AllReduce (2 GB/GPU, sum, float32)", collGB,
         avg(allReduceTimes, 0), avg(allReduceTimes, 1),
         minv(allReduceTimes, 1), maxv(allReduceTimes, 1));
  printf("  %-36s  data_moved %4.1f GB  avg(all) %7.2f ms  avg(skip1) %7.2f ms  min %7.2f  max %7.2f\n",
         "NCCL Broadcast (4 GB, root=GPU_IDS[0])", collGB,
         avg(broadcastTimes, 0), avg(broadcastTimes, 1),
         minv(broadcastTimes, 1), maxv(broadcastTimes, 1));
  printf("  %-36s  data_moved %4.1f GB  avg(all) %7.2f ms  avg(skip1) %7.2f ms  min %7.2f  max %7.2f\n",
         "NCCL AllGather (1 GB/GPU, float32)", collGB,
         avg(allGatherTimes, 0), avg(allGatherTimes, 1),
         minv(allGatherTimes, 1), maxv(allGatherTimes, 1));
  printf("================================================================\n");

  return 0;
}
