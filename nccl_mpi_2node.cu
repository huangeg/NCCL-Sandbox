// nccl_mpi_2node.cu
// Cross-node full-duplex NCCL send/recv benchmark with MPI coordination.
// Designed for AWS g7e.8xlarge (EFA + GPUDirect RDMA, aws-ofi-nccl plugin).
//
// Each rank picks its GPU via OMPI_COMM_WORLD_LOCAL_RANK (overridable with
// --local-gpu). Rank 0 generates the ncclUniqueId and broadcasts it to all
// peers via MPI_Bcast — no TCP rendezvous socket needed.
//
// Run pattern (2 ranks, 1 per node):
//   rank 0: ncclSend→1,  ncclRecv←1  (full-duplex group)
//   rank 1: ncclRecv←0,  ncclSend→0

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ── error macros ─────────────────────────────────────────────────────────── */
static int g_rank = -1;

#define CUDA_CHECK(cmd) do {                                              \
  cudaError_t e = (cmd);                                                  \
  if (e != cudaSuccess) {                                                  \
    fprintf(stderr, "[rank %d] CUDA error %s:%d: %s\n",                  \
            g_rank, __FILE__, __LINE__, cudaGetErrorString(e));           \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                              \
  }                                                                       \
} while(0)

#define NCCL_CHECK(cmd) do {                                              \
  ncclResult_t r = (cmd);                                                  \
  if (r != ncclSuccess) {                                                  \
    fprintf(stderr, "[rank %d] NCCL error %s:%d: %s\n",                  \
            g_rank, __FILE__, __LINE__, ncclGetErrorString(r));           \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                              \
  }                                                                       \
} while(0)

#define MPI_CHECK(cmd) do {                                               \
  int e = (cmd);                                                           \
  if (e != MPI_SUCCESS) {                                                  \
    char errstr[MPI_MAX_ERROR_STRING]; int len;                           \
    MPI_Error_string(e, errstr, &len);                                    \
    fprintf(stderr, "[rank %d] MPI error %s:%d: %s\n",                   \
            g_rank, __FILE__, __LINE__, errstr);                          \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                              \
  }                                                                       \
} while(0)

/* ── config ───────────────────────────────────────────────────────────────── */
struct Config {
  int  localGpu    = -1;           // -1 = auto from OMPI_COMM_WORLD_LOCAL_RANK
  size_t sizeBytes = 64ULL << 20;  // 64 MiB per direction
  int  warmupIters = 5;
  int  iters       = 20;
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: mpirun -np 2 --host <node0>:1,<node1>:1 %s [options]\n\n"
    "Options:\n"
    "  --local-gpu <id>   GPU index on this node (default: OMPI_COMM_WORLD_LOCAL_RANK)\n"
    "  --size-mb <int>    Payload MiB per direction (default: 64)\n"
    "  --warmup <int>     Warmup iterations (default: 5)\n"
    "  --iters <int>      Timed iterations (default: 20)\n\n"
    "Each iteration: full-duplex exchange (send+recv in one ncclGroup).\n"
    "Bandwidth = 2 × size / time (full-duplex).\n",
    prog);
}

static bool parseInt(const char* s, int* out) {
  char* end = nullptr;
  long v = strtol(s, &end, 10);
  if (!s || *s == '\0' || !end || *end != '\0') return false;
  if (v < INT32_MIN || v > INT32_MAX) return false;
  *out = static_cast<int>(v);
  return true;
}

static Config parseArgs(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--local-gpu") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.localGpu)) { usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    } else if (!strcmp(argv[i], "--size-mb") && i + 1 < argc) {
      int mb = 0;
      if (!parseInt(argv[++i], &mb) || mb <= 0) { usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
      c.sizeBytes = static_cast<size_t>(mb) << 20;
    } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.warmupIters) || c.warmupIters < 0) { usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.iters) || c.iters <= 0) { usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage(argv[0]);
      MPI_Finalize();
      exit(EXIT_SUCCESS);
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  return c;
}

/* ── main ─────────────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));

  int rank, nranks;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  g_rank = rank;

  if (nranks < 2) {
    if (rank == 0)
      fprintf(stderr, "Need at least 2 MPI ranks (got %d).\n", nranks);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Config cfg = parseArgs(argc, argv);

  // Determine which GPU this rank should use.
  // Default: local rank within the node (one GPU per rank).
  if (cfg.localGpu < 0) {
    cfg.localGpu = 0;
    const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (lr) cfg.localGpu = atoi(lr);
  }

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (cfg.localGpu >= deviceCount) {
    fprintf(stderr, "[rank %d] local GPU %d out of range (device count=%d)\n",
            rank, cfg.localGpu, deviceCount);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CUDA_CHECK(cudaSetDevice(cfg.localGpu));

  // Print node/GPU assignment.
  {
    char hostname[256]; hostname[0] = '\0';
    gethostname(hostname, sizeof(hostname));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.localGpu));
    // Serialize rank-0 print first, then others.
    for (int r = 0; r < nranks; ++r) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      if (r == rank) {
        printf("[rank %d] host=%-20s  GPU %d: %s\n",
               rank, hostname, cfg.localGpu, prop.name);
        fflush(stdout);
      }
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
      printf("\n");
      printf("══════════════════════════════════════════════════════════════\n");
      printf(" NCCL MPI Cross-Node Send/Recv Benchmark (EFA GPUDirect RDMA)\n");
      printf("══════════════════════════════════════════════════════════════\n");
      printf(" Ranks   : %d\n", nranks);
      printf(" Msg size: %.2f MiB per direction\n", cfg.sizeBytes / 1048576.0);
      printf(" Warmup  : %d   Iters: %d\n", cfg.warmupIters, cfg.iters);
      printf("══════════════════════════════════════════════════════════════\n\n");
      fflush(stdout);
    }
  }

  // ── Bootstrap NCCL via MPI ────────────────────────────────────────────────
  ncclUniqueId uid;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&uid));
  MPI_CHECK(MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, uid, rank));

  // ── Allocate device buffers ───────────────────────────────────────────────
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  CUDA_CHECK(cudaMalloc(&sendBuf, cfg.sizeBytes));
  CUDA_CHECK(cudaMalloc(&recvBuf, cfg.sizeBytes));
  CUDA_CHECK(cudaMemset(sendBuf, 0x3A + rank, cfg.sizeBytes));
  CUDA_CHECK(cudaMemset(recvBuf, 0, cfg.sizeBytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t startEv, stopEv;
  CUDA_CHECK(cudaEventCreate(&startEv));
  CUDA_CHECK(cudaEventCreate(&stopEv));

  // ── Benchmark loop ────────────────────────────────────────────────────────
  std::vector<float> timedMs;
  timedMs.reserve(static_cast<size_t>(cfg.iters));

  const int totalIters = cfg.warmupIters + cfg.iters;
  for (int it = 0; it < totalIters; ++it) {
    CUDA_CHECK(cudaEventRecord(startEv, stream));

    NCCL_CHECK(ncclGroupStart());
    if (rank == 0) {
      NCCL_CHECK(ncclSend(sendBuf, cfg.sizeBytes, ncclChar, 1, comm, stream));
      NCCL_CHECK(ncclRecv(recvBuf, cfg.sizeBytes, ncclChar, 1, comm, stream));
    } else {
      NCCL_CHECK(ncclRecv(recvBuf, cfg.sizeBytes, ncclChar, 0, comm, stream));
      NCCL_CHECK(ncclSend(sendBuf, cfg.sizeBytes, ncclChar, 0, comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());

    CUDA_CHECK(cudaEventRecord(stopEv, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEv));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startEv, stopEv));

    if (it >= cfg.warmupIters) {
      timedMs.push_back(ms);
      const double bytesMoved = static_cast<double>(cfg.sizeBytes) * 2.0;
      const double gbps = (bytesMoved / 1e9) / (static_cast<double>(ms) / 1e3);
      if (rank == 0) {
        printf("[rank %d] iter %d/%d: %.3f ms, %.3f GB/s (full-duplex)\n",
               rank, it - cfg.warmupIters + 1, cfg.iters, ms, gbps);
        fflush(stdout);
      }
    }
  }

  // ── Per-rank summary ──────────────────────────────────────────────────────
  double sumMs = 0.0, minMs = 1e30, maxMs = 0.0;
  for (float ms : timedMs) {
    double d = static_cast<double>(ms);
    sumMs += d;
    if (d < minMs) minMs = d;
    if (d > maxMs) maxMs = d;
  }
  double avgMs  = sumMs / static_cast<double>(timedMs.size());
  const double bytesMoved = static_cast<double>(cfg.sizeBytes) * 2.0;
  double avgGBps = (bytesMoved / 1e9) / (avgMs  / 1e3);
  double minGBps = (bytesMoved / 1e9) / (maxMs  / 1e3);
  double maxGBps = (bytesMoved / 1e9) / (minMs  / 1e3);

  // Rank 0 reduces global min/max latency via MPI for a cross-node summary.
  double globalMinMs, globalMaxMs, globalAvgMs;
  MPI_CHECK(MPI_Reduce(&minMs, &globalMinMs, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&maxMs, &globalMaxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&avgMs, &globalAvgMs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
  globalAvgMs /= static_cast<double>(nranks);

  // All ranks print their local summary (serialized).
  for (int r = 0; r < nranks; ++r) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (r == rank) {
      printf("\n[rank %d] Local Summary\n", rank);
      printf("  msg size          : %.2f MiB per direction\n", cfg.sizeBytes / 1048576.0);
      printf("  timed iterations  : %d\n", cfg.iters);
      printf("  avg latency       : %.3f ms\n", avgMs);
      printf("  min / max latency : %.3f / %.3f ms\n", minMs, maxMs);
      printf("  avg full-duplex bw: %.3f GB/s\n", avgGBps);
      printf("  bw range          : %.3f - %.3f GB/s\n", minGBps, maxGBps);
      fflush(stdout);
    }
  }
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (rank == 0) {
    double gAvgGBps = (bytesMoved / 1e9) / (globalAvgMs / 1e3);
    double gMinGBps = (bytesMoved / 1e9) / (globalMaxMs / 1e3);
    double gMaxGBps = (bytesMoved / 1e9) / (globalMinMs / 1e3);
    printf("\n══ Cross-Node Summary (across all ranks) ══\n");
    printf("  avg latency       : %.3f ms\n", globalAvgMs);
    printf("  best latency      : %.3f ms\n", globalMinMs);
    printf("  worst latency     : %.3f ms\n", globalMaxMs);
    printf("  avg full-duplex bw: %.3f GB/s\n", gAvgGBps);
    printf("  bw range          : %.3f - %.3f GB/s\n", gMinGBps, gMaxGBps);
    fflush(stdout);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
  CUDA_CHECK(cudaEventDestroy(startEv));
  CUDA_CHECK(cudaEventDestroy(stopEv));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(sendBuf));
  CUDA_CHECK(cudaFree(recvBuf));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_CHECK(MPI_Finalize());
  return 0;
}
