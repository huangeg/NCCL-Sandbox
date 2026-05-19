// nccl_rdma_bench.cu
// Cross-node NCCL GPUDirect RDMA collective benchmark for AWS g6.12xlarge
// (4x NVIDIA L4, EFA 100 Gbps, aws-ofi-nccl).
//
// Bootstrap: same TCP socket rendezvous as nccl_send_recv_2node.cu — no MPI.
//   Rank 0 generates the ncclUniqueId, listens on masterPort, and sends the
//   ID to each connecting non-0 rank. All ranks then call ncclCommInitRank.
//
// Launch: run_nccl_rdma_bench.sh SSHes into each node and starts one process
//   per GPU, assigning contiguous global ranks across nodes.
//
// Timing: CUDA events on every rank; rank 0 is the reporter since a
//   collective only completes after all peers have contributed.
//
// Collectives tested: AllReduce, AllGather, ReduceScatter, Broadcast
// Message sizes: 1 KiB → 1 GiB (powers of four)

#include <cuda_runtime.h>
#include <nccl.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ── error macros ─────────────────────────────────────────────────────── */
static int g_rank = -1;

#define CUDA_CHECK(cmd) do {                                          \
  cudaError_t e = (cmd);                                              \
  if (e != cudaSuccess) {                                             \
    fprintf(stderr, "[rank %d] CUDA error %s:%d: %s\n",              \
            g_rank, __FILE__, __LINE__, cudaGetErrorString(e));       \
    exit(EXIT_FAILURE);                                               \
  }                                                                   \
} while(0)

#define NCCL_CHECK(cmd) do {                                          \
  ncclResult_t r = (cmd);                                             \
  if (r != ncclSuccess) {                                             \
    fprintf(stderr, "[rank %d] NCCL error %s:%d: %s\n",              \
            g_rank, __FILE__, __LINE__, ncclGetErrorString(r));       \
    exit(EXIT_FAILURE);                                               \
  }                                                                   \
} while(0)

/* ── config ───────────────────────────────────────────────────────────── */
struct Config {
  int rank         = -1;
  int nranks       = 2;
  int localGpu     = 0;
  std::string masterAddr = "127.0.0.1";
  int masterPort   = 50051;
  int warmupIters  = 5;
  int iters        = 20;
  std::string csvPath;   // written by rank 0 only
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage:\n"
    "  %s --rank <n> --nranks <N> --local-gpu <id>\n"
    "     --master-addr <ip> --master-port <port> [options]\n\n"
    "Options:\n"
    "  --warmup <int>   Warmup iterations (default: 5)\n"
    "  --iters  <int>   Timed iterations  (default: 20)\n"
    "  --csv    <path>  Write results CSV (rank 0 only)\n\n"
    "Collectives: AllReduce, AllGather, ReduceScatter, Broadcast\n"
    "Sizes:       1 KiB .. 1 GiB\n",
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
    if      (!strcmp(argv[i], "--rank")        && i+1 < argc) { if (!parseInt(argv[++i], &c.rank))      { usage(argv[0]); exit(EXIT_FAILURE); } }
    else if (!strcmp(argv[i], "--nranks")      && i+1 < argc) { if (!parseInt(argv[++i], &c.nranks))    { usage(argv[0]); exit(EXIT_FAILURE); } }
    else if (!strcmp(argv[i], "--local-gpu")   && i+1 < argc) { if (!parseInt(argv[++i], &c.localGpu))  { usage(argv[0]); exit(EXIT_FAILURE); } }
    else if (!strcmp(argv[i], "--master-addr") && i+1 < argc) { c.masterAddr = argv[++i]; }
    else if (!strcmp(argv[i], "--master-port") && i+1 < argc) { if (!parseInt(argv[++i], &c.masterPort)) { usage(argv[0]); exit(EXIT_FAILURE); } }
    else if (!strcmp(argv[i], "--warmup")      && i+1 < argc) { if (!parseInt(argv[++i], &c.warmupIters)) { usage(argv[0]); exit(EXIT_FAILURE); } }
    else if (!strcmp(argv[i], "--iters")       && i+1 < argc) { if (!parseInt(argv[++i], &c.iters))     { usage(argv[0]); exit(EXIT_FAILURE); } }
    else if (!strcmp(argv[i], "--csv")         && i+1 < argc) { c.csvPath = argv[++i]; }
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); exit(EXIT_SUCCESS); }
    else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); exit(EXIT_FAILURE); }
  }
  if (c.rank < 0 || c.rank >= c.nranks || c.nranks < 1 ||
      c.localGpu < 0 || c.masterPort <= 0 || c.masterPort > 65535 ||
      c.warmupIters < 0 || c.iters <= 0) {
    usage(argv[0]); exit(EXIT_FAILURE);
  }
  return c;
}

/* ── TCP socket helpers (same as nccl_send_recv_2node.cu) ─────────────── */
static void sendAll(int fd, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (n > 0) {
    ssize_t s = send(fd, p, n, 0);
    if (s < 0) { if (errno == EINTR) continue; perror("send"); exit(EXIT_FAILURE); }
    if (s == 0) { fprintf(stderr, "send returned 0\n"); exit(EXIT_FAILURE); }
    p += static_cast<size_t>(s); n -= static_cast<size_t>(s);
  }
}

static void recvAll(int fd, void* data, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (n > 0) {
    ssize_t r = recv(fd, p, n, 0);
    if (r < 0) { if (errno == EINTR) continue; perror("recv"); exit(EXIT_FAILURE); }
    if (r == 0) { fprintf(stderr, "peer closed socket early\n"); exit(EXIT_FAILURE); }
    p += static_cast<size_t>(r); n -= static_cast<size_t>(r);
  }
}

static int connectTo(const std::string& host, int port) {
  char portStr[16];
  snprintf(portStr, sizeof(portStr), "%d", port);
  struct addrinfo hints; memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), portStr, &hints, &res) != 0) return -1;
  int fd = -1;
  for (struct addrinfo* rp = res; rp; rp = rp->ai_next) {
    fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) { freeaddrinfo(res); return fd; }
    close(fd); fd = -1;
  }
  freeaddrinfo(res); return -1;
}

static int connectWithRetry(const std::string& host, int port, int retries) {
  for (int i = 0; i < retries; ++i) {
    int fd = connectTo(host, port);
    if (fd >= 0) return fd;
    usleep(200 * 1000);
  }
  return -1;
}

/* ── N-rank bootstrap ─────────────────────────────────────────────────── */
// Rank 0 generates the ncclUniqueId, listens, and sends it to each other rank.
// All other ranks connect to rank 0 and receive the ID.
// The connection backlog is set to nranks so all peers can queue before accept.
static void bootstrapNcclId(const Config& cfg, ncclUniqueId* uid) {
  if (cfg.rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(uid));

    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) { perror("socket"); exit(EXIT_FAILURE); }
    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(cfg.masterPort));
    if (bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) { perror("bind"); exit(EXIT_FAILURE); }
    if (listen(sfd, cfg.nranks) < 0) { perror("listen"); exit(EXIT_FAILURE); }

    for (int r = 1; r < cfg.nranks; ++r) {
      int cfd = accept(sfd, nullptr, nullptr);
      if (cfd < 0) { perror("accept"); exit(EXIT_FAILURE); }
      sendAll(cfd, uid, sizeof(*uid));
      close(cfd);
    }
    close(sfd);
  } else {
    int fd = connectWithRetry(cfg.masterAddr, cfg.masterPort, 200);
    if (fd < 0) {
      fprintf(stderr, "[rank %d] failed to connect to rank 0 bootstrap at %s:%d\n",
              cfg.rank, cfg.masterAddr.c_str(), cfg.masterPort);
      exit(EXIT_FAILURE);
    }
    recvAll(fd, uid, sizeof(*uid));
    close(fd);
  }
}

/* ── bandwidth helpers ────────────────────────────────────────────────── */
// Bus bandwidth = data transferred over network links / time.
// Factors follow nccl-tests convention.
//   AllReduce     : algBW * 2*(n-1)/n
//   AllGather     : algBW * (n-1)/n   (algBW computed on total recv'd bytes)
//   ReduceScatter : algBW * (n-1)/n
//   Broadcast     : algBW * (n-1)/n

static const char* opNames[]   = {"AllReduce", "AllGather", "ReduceScatter", "Broadcast"};
static const int   NUM_OPS     = 4;

static void fmtBytes(char* buf, size_t sz) {
  if      (sz >= (1UL<<30)) snprintf(buf, 16, "%.1f GiB", (double)sz/(1UL<<30));
  else if (sz >= (1UL<<20)) snprintf(buf, 16, "%.1f MiB", (double)sz/(1UL<<20));
  else if (sz >= (1UL<<10)) snprintf(buf, 16, "%.1f KiB", (double)sz/(1UL<<10));
  else                      snprintf(buf, 16, "%zu B",    sz);
}

/* ── single collective, single size ──────────────────────────────────── */
// Returns elapsed milliseconds on this rank for the timed section.
// algBW and busBW are filled for rank 0 (other ranks' values are discarded).
static float benchOne(int op, size_t msgBytes, int nranks,
                      ncclComm_t comm, cudaStream_t stream,
                      float* dSend, float* dRecv,
                      int warmup, int iters,
                      double* algBW_GBps, double* busBW_GBps)
{
  int n = nranks;
  size_t count = msgBytes / sizeof(float);

  // ReduceScatter requires count divisible by nranks
  if (op == 2 /* ReduceScatter */) {
    count = (count / n) * n;
    if (count == 0) count = static_cast<size_t>(n);
    msgBytes = count * sizeof(float);
  }
  if (count == 0) count = 1;

  auto runColl = [&]() {
    switch (op) {
    case 0: // AllReduce
      NCCL_CHECK(ncclAllReduce(dSend, dRecv, count, ncclFloat, ncclSum, comm, stream));
      break;
    case 1: // AllGather — send count floats, recv count*nranks floats
      NCCL_CHECK(ncclAllGather(dSend, dRecv, count, ncclFloat, comm, stream));
      break;
    case 2: // ReduceScatter — recv count/nranks floats
      NCCL_CHECK(ncclReduceScatter(dSend, dRecv, count / static_cast<size_t>(n),
                                    ncclFloat, ncclSum, comm, stream));
      break;
    case 3: // Broadcast from root 0
      NCCL_CHECK(ncclBcast(dSend, count, ncclFloat, 0, comm, stream));
      break;
    }
  };

  // Warmup
  for (int i = 0; i < warmup; ++i) runColl();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Timed iterations
  cudaEvent_t startEv, stopEv;
  CUDA_CHECK(cudaEventCreate(&startEv));
  CUDA_CHECK(cudaEventCreate(&stopEv));

  CUDA_CHECK(cudaEventRecord(startEv, stream));
  for (int i = 0; i < iters; ++i) runColl();
  CUDA_CHECK(cudaEventRecord(stopEv, stream));
  CUDA_CHECK(cudaEventSynchronize(stopEv));

  float totalMs = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&totalMs, startEv, stopEv));
  float perIterMs = totalMs / static_cast<float>(iters);

  CUDA_CHECK(cudaEventDestroy(startEv));
  CUDA_CHECK(cudaEventDestroy(stopEv));

  // Bandwidth (bytes / ms → GB/s: divide by 1e6)
  double dataBytes;
  double busFactor;
  switch (op) {
  case 0: dataBytes = (double)msgBytes;             busFactor = 2.0 * (n-1) / n; break; // AllReduce
  case 1: dataBytes = (double)msgBytes * n;         busFactor = (double)(n-1) / n; break; // AllGather
  case 2: dataBytes = (double)msgBytes;             busFactor = (double)(n-1) / n; break; // ReduceScatter
  case 3: dataBytes = (double)msgBytes;             busFactor = (double)(n-1) / n; break; // Broadcast
  default: dataBytes = (double)msgBytes;            busFactor = 1.0; break;
  }
  *algBW_GBps = dataBytes / static_cast<double>(perIterMs) / 1e6;
  *busBW_GBps = (*algBW_GBps) * busFactor;

  return perIterMs;
}

/* ── full sweep ───────────────────────────────────────────────────────── */
static void benchAll(const Config& cfg,
                     ncclComm_t comm, cudaStream_t stream,
                     float* dSend, float* dRecv)
{
  static const size_t SIZES[] = {
    1UL<<10, 1UL<<12, 1UL<<14, 1UL<<16,
    1UL<<18, 1UL<<20, 1UL<<22, 1UL<<24,
    1UL<<26, 1UL<<28, 1UL<<30
  };
  const int NSIZES = static_cast<int>(sizeof(SIZES) / sizeof(SIZES[0]));

  FILE* csv = nullptr;
  if (cfg.rank == 0 && !cfg.csvPath.empty()) {
    csv = fopen(cfg.csvPath.c_str(), "w");
    if (!csv) fprintf(stderr, "[rank 0] warning: cannot open CSV %s\n", cfg.csvPath.c_str());
    else fprintf(csv, "op,msg_bytes,lat_ms,algBW_GBps,busBW_GBps\n");
  }

  for (int op = 0; op < NUM_OPS; ++op) {
    if (cfg.rank == 0) {
      printf("┌─────────────────────────────────────────────────────────────┐\n");
      printf("│ %-61s│\n", opNames[op]);
      printf("├────────────┬──────────────┬──────────────┬──────────────────┤\n");
      printf("│ Msg Size   │  Lat (ms)    │  AlgBW GB/s  │  BusBW GB/s      │\n");
      printf("├────────────┼──────────────┼──────────────┼──────────────────┤\n");
      fflush(stdout);
    }

    for (int si = 0; si < NSIZES; ++si) {
      size_t sz = SIZES[si];
      // AllGather recv buffer must be nranks * send size — skip if it overflows
      size_t recvNeed = (op == 1) ? sz * static_cast<size_t>(cfg.nranks) : sz;
      // We allocated dRecv as MAX_SIZE * nranks bytes; check it fits
      static const size_t MAX_SIZE = 1UL << 30;
      if (sz > MAX_SIZE || recvNeed > MAX_SIZE * static_cast<size_t>(cfg.nranks)) break;

      double algBW = 0.0, busBW = 0.0;
      float latMs = benchOne(op, sz, cfg.nranks, comm, stream,
                              dSend, dRecv,
                              cfg.warmupIters, cfg.iters,
                              &algBW, &busBW);

      if (cfg.rank == 0) {
        char sbuf[16]; fmtBytes(sbuf, sz);
        printf("│ %-10s │ %12.3f │ %12.3f │ %16.3f │\n",
               sbuf, (double)latMs, algBW, busBW);
        fflush(stdout);
        if (csv) fprintf(csv, "%s,%zu,%.4f,%.4f,%.4f\n",
                         opNames[op], sz, (double)latMs, algBW, busBW);
      }
    }

    if (cfg.rank == 0) {
      printf("└────────────┴──────────────┴──────────────┴──────────────────┘\n\n");
      fflush(stdout);
    }
  }

  if (csv) fclose(csv);
}

/* ── main ─────────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
  Config cfg = parseArgs(argc, argv);
  g_rank = cfg.rank;

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (cfg.localGpu >= deviceCount) {
    fprintf(stderr, "[rank %d] local GPU %d out of range (count=%d)\n",
            cfg.rank, cfg.localGpu, deviceCount);
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(cfg.localGpu));

  // Bootstrap NCCL communicator via TCP (no MPI)
  ncclUniqueId uid;
  bootstrapNcclId(cfg, &uid);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, cfg.nranks, uid, cfg.rank));

  // Allocate device buffers
  // dRecv is nranks × MAX_SIZE to accommodate AllGather output
  static const size_t MAX_SIZE = 1UL << 30;  // 1 GiB
  float* dSend = nullptr;
  float* dRecv = nullptr;
  CUDA_CHECK(cudaMalloc(&dSend, MAX_SIZE));
  CUDA_CHECK(cudaMalloc(&dRecv, MAX_SIZE * static_cast<size_t>(cfg.nranks)));
  CUDA_CHECK(cudaMemset(dSend, 0, MAX_SIZE));
  CUDA_CHECK(cudaMemset(dRecv, 0, MAX_SIZE * static_cast<size_t>(cfg.nranks)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  if (cfg.rank == 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.localGpu));
    printf("══════════════════════════════════════════════════════════════\n");
    printf(" NCCL GPUDirect RDMA Benchmark\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf(" Ranks    : %d total\n", cfg.nranks);
    printf(" GPU[r=0] : %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    printf(" Warmup   : %d   Iters: %d\n", cfg.warmupIters, cfg.iters);
    if (!cfg.csvPath.empty()) printf(" CSV out  : %s\n", cfg.csvPath.c_str());
    printf("══════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);
  }

  benchAll(cfg, comm, stream, dSend, dRecv);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dSend));
  CUDA_CHECK(cudaFree(dRecv));
  NCCL_CHECK(ncclCommDestroy(comm));
  return 0;
}
