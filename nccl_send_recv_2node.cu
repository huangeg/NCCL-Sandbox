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

struct Config {
  int rank = -1;
  int localGpu = 0;
  std::string masterAddr = "127.0.0.1";
  int masterPort = 50050;
  size_t sizeBytes = 64ULL << 20;  // 64 MiB
  int warmupIters = 5;
  int iters = 20;
};

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s --rank <0|1> --local-gpu <id> --master-addr <host/ip> --master-port <port> [options]\n\n"
          "Options:\n"
          "  --size-mb <int>     Payload size in MiB per direction (default: 64)\n"
          "  --warmup <int>      Warmup iterations (default: 5)\n"
          "  --iters <int>       Timed iterations (default: 20)\n"
          "\n"
          "Each iteration performs one full-duplex exchange:\n"
          "  rank0: send->1, recv<-1\n"
          "  rank1: recv<-0, send->0\n"
          "Bandwidth reported as aggregate full-duplex GB/s.\n",
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

static bool parseSizeMB(const char* s, size_t* outBytes) {
  int mb = 0;
  if (!parseInt(s, &mb) || mb <= 0) return false;
  *outBytes = static_cast<size_t>(mb) << 20;
  return true;
}

static Config parseArgs(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rank") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.rank)) { usage(argv[0]); exit(EXIT_FAILURE); }
    } else if (!strcmp(argv[i], "--local-gpu") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.localGpu)) { usage(argv[0]); exit(EXIT_FAILURE); }
    } else if (!strcmp(argv[i], "--master-addr") && i + 1 < argc) {
      c.masterAddr = argv[++i];
    } else if (!strcmp(argv[i], "--master-port") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.masterPort)) { usage(argv[0]); exit(EXIT_FAILURE); }
    } else if (!strcmp(argv[i], "--size-mb") && i + 1 < argc) {
      if (!parseSizeMB(argv[++i], &c.sizeBytes)) { usage(argv[0]); exit(EXIT_FAILURE); }
    } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.warmupIters)) { usage(argv[0]); exit(EXIT_FAILURE); }
    } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
      if (!parseInt(argv[++i], &c.iters)) { usage(argv[0]); exit(EXIT_FAILURE); }
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    } else {
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if ((c.rank != 0 && c.rank != 1) || c.masterPort <= 0 || c.masterPort > 65535 ||
      c.localGpu < 0 || c.warmupIters < 0 || c.iters <= 0 || c.sizeBytes == 0) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  return c;
}

static void sendAll(int fd, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (n > 0) {
    ssize_t s = send(fd, p, n, 0);
    if (s < 0) {
      if (errno == EINTR) continue;
      perror("send");
      exit(EXIT_FAILURE);
    }
    if (s == 0) {
      fprintf(stderr, "send returned 0\n");
      exit(EXIT_FAILURE);
    }
    p += static_cast<size_t>(s);
    n -= static_cast<size_t>(s);
  }
}

static void recvAll(int fd, void* data, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (n > 0) {
    ssize_t r = recv(fd, p, n, 0);
    if (r < 0) {
      if (errno == EINTR) continue;
      perror("recv");
      exit(EXIT_FAILURE);
    }
    if (r == 0) {
      fprintf(stderr, "peer closed socket early\n");
      exit(EXIT_FAILURE);
    }
    p += static_cast<size_t>(r);
    n -= static_cast<size_t>(r);
  }
}

static int connectTo(const std::string& host, int port) {
  char portStr[16];
  snprintf(portStr, sizeof(portStr), "%d", port);

  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* res = nullptr;
  int gai = getaddrinfo(host.c_str(), portStr, &hints, &res);
  if (gai != 0) {
    fprintf(stderr, "getaddrinfo failed for %s:%d: %s\n", host.c_str(), port, gai_strerror(gai));
    exit(EXIT_FAILURE);
  }

  int fd = -1;
  for (struct addrinfo* rp = res; rp != nullptr; rp = rp->ai_next) {
    fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) continue;

    if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
      freeaddrinfo(res);
      return fd;
    }

    close(fd);
    fd = -1;
  }

  freeaddrinfo(res);
  return -1;
}

static int connectWithRetry(const std::string& host, int port, int retries) {
  for (int i = 0; i < retries; ++i) {
    int fd = connectTo(host, port);
    if (fd >= 0) return fd;
    usleep(200 * 1000);  // 200ms
  }
  return -1;
}

static int listenAndAccept(int port) {
  int serverFd = socket(AF_INET, SOCK_STREAM, 0);
  if (serverFd < 0) {
    perror("socket");
    exit(EXIT_FAILURE);
  }

  int opt = 1;
  if (setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
    perror("setsockopt");
    close(serverFd);
    exit(EXIT_FAILURE);
  }

  sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));

  if (bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    perror("bind");
    close(serverFd);
    exit(EXIT_FAILURE);
  }

  if (listen(serverFd, 1) != 0) {
    perror("listen");
    close(serverFd);
    exit(EXIT_FAILURE);
  }

  int connFd = accept(serverFd, nullptr, nullptr);
  if (connFd < 0) {
    perror("accept");
    close(serverFd);
    exit(EXIT_FAILURE);
  }

  close(serverFd);
  return connFd;
}

static void bootstrapNcclId(const Config& cfg, ncclUniqueId* outId) {
  if (cfg.rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(outId));
    int fd = listenAndAccept(cfg.masterPort);
    sendAll(fd, outId, sizeof(*outId));
    close(fd);
  } else {
    int fd = connectWithRetry(cfg.masterAddr, cfg.masterPort, 100);
    if (fd < 0) {
      fprintf(stderr, "failed to connect to rank0 bootstrap at %s:%d\n",
              cfg.masterAddr.c_str(), cfg.masterPort);
      exit(EXIT_FAILURE);
    }
    recvAll(fd, outId, sizeof(*outId));
    close(fd);
  }
}

int main(int argc, char** argv) {
  Config cfg = parseArgs(argc, argv);

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (cfg.localGpu >= deviceCount) {
    fprintf(stderr, "local GPU %d out of range (device count=%d)\n", cfg.localGpu, deviceCount);
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cudaSetDevice(cfg.localGpu));

  ncclUniqueId uid;
  bootstrapNcclId(cfg, &uid);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, 2, uid, cfg.rank));

  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  CUDA_CHECK(cudaMalloc(&sendBuf, cfg.sizeBytes));
  CUDA_CHECK(cudaMalloc(&recvBuf, cfg.sizeBytes));
  CUDA_CHECK(cudaMemset(sendBuf, 0x3A + cfg.rank, cfg.sizeBytes));
  CUDA_CHECK(cudaMemset(recvBuf, 0, cfg.sizeBytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t startEv;
  cudaEvent_t stopEv;
  CUDA_CHECK(cudaEventCreate(&startEv));
  CUDA_CHECK(cudaEventCreate(&stopEv));

  std::vector<float> timedMs;
  timedMs.reserve(static_cast<size_t>(cfg.iters));

  const int totalIters = cfg.warmupIters + cfg.iters;
  for (int it = 0; it < totalIters; ++it) {
    CUDA_CHECK(cudaEventRecord(startEv, stream));

    NCCL_CHECK(ncclGroupStart());
    if (cfg.rank == 0) {
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
      const double bytesMoved = static_cast<double>(cfg.sizeBytes) * 2.0;  // full duplex
      const double gbps = (bytesMoved / 1e9) / (static_cast<double>(ms) / 1e3);
      printf("[rank %d] iter %d/%d: %.3f ms, %.3f GB/s (full-duplex)\n",
             cfg.rank, it - cfg.warmupIters + 1, cfg.iters, ms, gbps);
      fflush(stdout);
    }
  }

  double sumMs = 0.0;
  double minMs = 1e30;
  double maxMs = 0.0;
  for (float ms : timedMs) {
    double d = static_cast<double>(ms);
    sumMs += d;
    if (d < minMs) minMs = d;
    if (d > maxMs) maxMs = d;
  }
  double avgMs = sumMs / static_cast<double>(timedMs.size());

  const double bytesMoved = static_cast<double>(cfg.sizeBytes) * 2.0;
  double avgGBps = (bytesMoved / 1e9) / (avgMs / 1e3);
  double minGBps = (bytesMoved / 1e9) / (maxMs / 1e3);
  double maxGBps = (bytesMoved / 1e9) / (minMs / 1e3);

  printf("\n[rank %d] Summary\n", cfg.rank);
  printf("  message size        : %.2f MiB per direction\n", cfg.sizeBytes / 1048576.0);
  printf("  timed iterations    : %d\n", cfg.iters);
  printf("  avg latency         : %.3f ms\n", avgMs);
  printf("  min latency         : %.3f ms\n", minMs);
  printf("  max latency         : %.3f ms\n", maxMs);
  printf("  avg full-duplex bw  : %.3f GB/s\n", avgGBps);
  printf("  bw range            : %.3f - %.3f GB/s\n", minGBps, maxGBps);
  fflush(stdout);

  CUDA_CHECK(cudaEventDestroy(startEv));
  CUDA_CHECK(cudaEventDestroy(stopEv));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(sendBuf));
  CUDA_CHECK(cudaFree(recvBuf));
  NCCL_CHECK(ncclCommDestroy(comm));

  return 0;
}
