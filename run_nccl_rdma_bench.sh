#!/usr/bin/env bash
# run_nccl_rdma_bench.sh
# Cross-node NCCL GPUDirect RDMA collective benchmark launcher.
#
# Mirrors run_nccl_send_recv_2node.sh in structure:
#   1) Builds nccl_rdma_bench on both nodes via SSH (or locally for node0).
#   2) Launches one process per GPU on each node, with contiguous global ranks:
#        node0 → ranks 0 .. GPUS_PER_NODE-1
#        node1 → ranks GPUS_PER_NODE .. 2*GPUS_PER_NODE-1
#   3) Rank 0 acts as the TCP bootstrap server; the script waits for its
#      port to open before starting node1's processes (same as 2-node script).
#   4) EFA + GPUDirect RDMA env vars are exported to every process.
#
# Usage:
#   ./run_nccl_rdma_bench.sh \
#     --node0 user@host0 \
#     --node1 user@host1 \
#     --node0-ip <ip-reachable-from-node1> \
#     [--project-dir <abs-path-on-both-nodes>]
#     [--project-dir0 <path-on-node0> --project-dir1 <path-on-node1>]
#     [--gpus-per-node 4] [--warmup 5] [--iters 20] [--port 50051]
#     [--iface eth0] [--csv results.csv]

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_nccl_rdma_bench.sh \
    --node0 user@host0 \
    --node1 user@host1 \
    --node0-ip <ip-reachable-from-node1> \
    [--project-dir <absolute-path-on-both-nodes>
     | --project-dir0 <path> --project-dir1 <path>] \
    [--gpus-per-node 4] [--warmup 5] [--iters 20] \
    [--port 50051] [--iface eth0] [--csv <file>]

Required:
  --node0       SSH target for rank-0 node, e.g. ubuntu@10.0.1.10
  --node1       SSH target for rank-1 node, e.g. ubuntu@10.0.1.11
  --node0-ip    IP/hostname that node1 can reach for NCCL bootstrap

Path (choose one):
  --project-dir   Absolute path of this repo on BOTH nodes
  --project-dir0  Path on node0    --project-dir1  Path on node1

Notes:
  - node0 can be the local machine; it is detected and run without SSH.
  - node1 requires passwordless SSH.
  - Default gpus-per-node is 4 (matches g6.12xlarge with 4x L4 GPUs).
  - Each GPU gets one process; global rank = node_offset + local_gpu_id.
  - EFA GPUDirect RDMA is enabled via FI_EFA_USE_DEVICE_RDMA=1.
  - Adjust --iface to your EFA/Ethernet NIC (e.g. ens5, bond0).
EOF
}

# ── defaults ──────────────────────────────────────────────────────────────
NODE0=""
NODE1=""
NODE0_IP=""
PROJECT_DIR=""
PROJECT_DIR0=""
PROJECT_DIR1=""
GPUS_PER_NODE=4
WARMUP=5
ITERS=20
PORT=50051
IFACE="eth0"
CSV_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node0)         NODE0="$2";         shift 2 ;;
    --node1)         NODE1="$2";         shift 2 ;;
    --node0-ip)      NODE0_IP="$2";      shift 2 ;;
    --project-dir)   PROJECT_DIR="$2";   shift 2 ;;
    --project-dir0)  PROJECT_DIR0="$2";  shift 2 ;;
    --project-dir1)  PROJECT_DIR1="$2";  shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --warmup)        WARMUP="$2";        shift 2 ;;
    --iters)         ITERS="$2";         shift 2 ;;
    --port)          PORT="$2";          shift 2 ;;
    --iface)         IFACE="$2";         shift 2 ;;
    --csv)           CSV_ARG="--csv $2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$NODE0" || -z "$NODE1" || -z "$NODE0_IP" ]]; then
  echo "ERROR: --node0, --node1, and --node0-ip are required." >&2; usage; exit 1
fi

if [[ -n "$PROJECT_DIR" && ( -n "$PROJECT_DIR0" || -n "$PROJECT_DIR1" ) ]]; then
  echo "ERROR: Use --project-dir OR (--project-dir0 + --project-dir1), not both." >&2; usage; exit 1
fi
[[ -n "$PROJECT_DIR" ]] && { PROJECT_DIR0="$PROJECT_DIR"; PROJECT_DIR1="$PROJECT_DIR"; }
if [[ -z "$PROJECT_DIR0" || -z "$PROJECT_DIR1" ]]; then
  echo "ERROR: Missing project path. Provide --project-dir or both --project-dir0/1." >&2; usage; exit 1
fi

NRANKS=$(( GPUS_PER_NODE * 2 ))

# ── detect if node0 is local (skip SSH for rank 0 node) ──────────────────
node0_host="${NODE0#*@}"
local_host="$(hostname)"
local_host_s="$(hostname -s)"
node0_is_local=false
if [[ "$node0_host" == "$local_host"   || "$node0_host" == "$local_host_s" ||
      "$node0_host" == "localhost"      || "$node0_host" == "127.0.0.1" ]]; then
  node0_is_local=true
  echo "Detected node0 ($NODE0) is the local machine; node0 processes run directly."
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR0="$PROJECT_DIR0/logs"
LOG_DIR1="$PROJECT_DIR1/logs"

# ── library paths propagated to all ranks ────────────────────────────────
CUDA_HOME=/opt/pytorch/cuda
OFI_LIB=/opt/amazon/ofi-nccl/lib
EFA_LIB=/opt/amazon/efa/lib
MPI_LIB=/opt/amazon/openmpi/lib
LDPATH="${OFI_LIB}:${CUDA_HOME}/lib:${EFA_LIB}:${MPI_LIB}"

# ── build commands ────────────────────────────────────────────────────────
BUILD_CMD0="cd '$PROJECT_DIR0' && make nccl_rdma_bench"
BUILD_CMD1="cd '$PROJECT_DIR1' && make nccl_rdma_bench"

# ── per-node launch: start GPUS_PER_NODE background processes ────────────
# Each process logs to logs/nccl_rdma_bench_rank<N>_<TS>.log
# The outer shell waits for all of them.
make_run_cmd() {
  local dir="$1"
  local node_rank_offset="$2"   # 0 for node0, GPUS_PER_NODE for node1
  local log_dir="$3"
  local is_rank0_node="$4"      # "true" if this node hosts global rank 0

  cat <<CMD
cd '$dir' && mkdir -p '$log_dir'
for i in \$(seq 0 $((GPUS_PER_NODE - 1))); do
  GLOBAL_RANK=\$((${node_rank_offset} + i))
  LD_LIBRARY_PATH='${LDPATH}' \\
  FI_PROVIDER=efa \\
  FI_EFA_USE_DEVICE_RDMA=1 \\
  FI_EFA_FORK_SAFE=1 \\
  RDMAV_FORK_SAFE=1 \\
  NCCL_NET_GDR_LEVEL=5 \\
  NCCL_NET_GDR_READ=1 \\
  NCCL_ALGO=Ring \\
  NCCL_PROTO=Simple \\
  NCCL_SOCKET_IFNAME='${IFACE}' \\
  NCCL_DEBUG=WARN \\
  ./nccl_rdma_bench \\
    --rank \$GLOBAL_RANK \\
    --nranks ${NRANKS} \\
    --local-gpu \$i \\
    --master-addr '${NODE0_IP}' \\
    --master-port ${PORT} \\
    --warmup ${WARMUP} \\
    --iters ${ITERS} \\
    ${CSV_ARG} \\
    2>&1 | tee '${log_dir}/nccl_rdma_bench_rank'\$GLOBAL_RANK'_${TS}.log' &
done
wait
CMD
}

RUN_NODE0_CMD="$(make_run_cmd "$PROJECT_DIR0" 0            "$LOG_DIR0" true)"
RUN_NODE1_CMD="$(make_run_cmd "$PROJECT_DIR1" "$GPUS_PER_NODE" "$LOG_DIR1" false)"

# ── build ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo " NCCL RDMA Benchmark — $NRANKS ranks across 2 nodes"
echo "══════════════════════════════════════════════════════════"
echo " node0     : $NODE0  ($GPUS_PER_NODE GPUs, ranks 0-$((GPUS_PER_NODE-1)))"
echo " node1     : $NODE1  ($GPUS_PER_NODE GPUs, ranks $GPUS_PER_NODE-$((NRANKS-1)))"
echo " node0-ip  : $NODE0_IP (bootstrap addr for all ranks)"
echo " port      : $PORT"
echo " iface     : $IFACE"
echo " warmup/iters: $WARMUP / $ITERS"
[[ -n "$CSV_ARG" ]] && echo " csv       : $(echo "$CSV_ARG" | awk '{print $2}')"
echo "══════════════════════════════════════════════════════════"

echo ""
echo "== Building on node0: $NODE0 =="
if $node0_is_local; then
  eval "$BUILD_CMD0"
else
  ssh "$NODE0" "$BUILD_CMD0"
fi

echo "== Building on node1: $NODE1 =="
ssh "$NODE1" "$BUILD_CMD1"

# ── launch node0 processes (includes global rank 0, the bootstrap server) ─
echo ""
echo "== Launching $GPUS_PER_NODE rank(s) on node0: $NODE0 =="
mkdir -p "$LOG_DIR0"
if $node0_is_local; then
  eval "$RUN_NODE0_CMD" &
else
  ssh "$NODE0" "$RUN_NODE0_CMD" &
fi
PID0=$!

# Wait until rank 0's bootstrap port is open before starting node1
echo "Waiting for rank 0 to open bootstrap port $PORT ..."
for i in $(seq 1 60); do
  sleep 1
  if $node0_is_local; then
    ss -ltn 2>/dev/null | grep -q ":${PORT}" && break
  else
    ssh "$NODE0" "ss -ltn 2>/dev/null | grep -q ':${PORT}'" 2>/dev/null && break
  fi
done
echo "Rank 0 bootstrap is ready."

# ── launch node1 processes ────────────────────────────────────────────────
echo "== Launching $GPUS_PER_NODE rank(s) on node1: $NODE1 =="
ssh "$NODE1" "$RUN_NODE1_CMD" &
PID1=$!

# ── wait and report ───────────────────────────────────────────────────────
set +e
wait "$PID0"; RC0=$?
wait "$PID1"; RC1=$?
set -e

echo ""
echo "Node0 exit code: $RC0"
echo "Node1 exit code: $RC1"
echo "Logs:"
echo "  ${NODE0}:${LOG_DIR0}/nccl_rdma_bench_rank*_${TS}.log"
echo "  ${NODE1}:${LOG_DIR1}/nccl_rdma_bench_rank*_${TS}.log"

if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then exit 1; fi
echo "Cross-node NCCL RDMA benchmark completed."
