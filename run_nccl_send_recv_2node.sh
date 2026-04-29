#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_nccl_send_recv_2node.sh \
    --node0 user@host0 \
    --node1 user@host1 \
    --node0-ip <ip-reachable-from-node1> \
    [--project-dir <absolute-path-on-both-nodes> | --project-dir0 <path-on-node0> --project-dir1 <path-on-node1>] \
    [--gpu0 0] [--gpu1 0] [--size-mb 64] [--warmup 5] [--iters 20] [--port 50050] [--iface eth0]

Description:
  1) Compiles nccl_send_recv_2node.cu on both nodes.
  2) Starts rank 0 on node0 and rank 1 on node1.
  3) Forces NCCL to use socket/Ethernet transport.
  4) Saves logs under ./logs on node0 (and temporary rank1 log on node1).

Required:
  --node0       SSH target for rank 0, e.g. ubuntu@10.0.0.11
  --node1       SSH target for rank 1, e.g. ubuntu@10.0.0.12
  --node0-ip    IP/hostname node1 can reach for NCCL bootstrap socket
  Path options (choose one mode):
  - --project-dir  Absolute path where this repo exists on BOTH nodes
  - --project-dir0 Absolute path for node0 and --project-dir1 for node1

Notes:
  - Assumes passwordless SSH is configured.
  - If your NIC is not eth0, pass --iface <name> (example: ens3).
EOF
}

NODE0=""
NODE1=""
NODE0_IP=""
PROJECT_DIR=""
PROJECT_DIR0=""
PROJECT_DIR1=""
GPU0=0
GPU1=0
SIZE_MB=64
WARMUP=5
ITERS=20
PORT=50050
IFACE="eth0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node0) NODE0="$2"; shift 2 ;;
    --node1) NODE1="$2"; shift 2 ;;
    --node0-ip) NODE0_IP="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --project-dir0) PROJECT_DIR0="$2"; shift 2 ;;
    --project-dir1) PROJECT_DIR1="$2"; shift 2 ;;
    --gpu0) GPU0="$2"; shift 2 ;;
    --gpu1) GPU1="$2"; shift 2 ;;
    --size-mb) SIZE_MB="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --iface) IFACE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$NODE0" || -z "$NODE1" || -z "$NODE0_IP" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ -n "$PROJECT_DIR" && ( -n "$PROJECT_DIR0" || -n "$PROJECT_DIR1" ) ]]; then
  echo "Use either --project-dir OR (--project-dir0 and --project-dir1), not both." >&2
  usage
  exit 1
fi

if [[ -n "$PROJECT_DIR" ]]; then
  PROJECT_DIR0="$PROJECT_DIR"
  PROJECT_DIR1="$PROJECT_DIR"
fi

if [[ -z "$PROJECT_DIR0" || -z "$PROJECT_DIR1" ]]; then
  echo "Missing project path arguments. Provide --project-dir or both --project-dir0 and --project-dir1." >&2
  usage
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR0="$PROJECT_DIR0/logs"
LOG_DIR1="$PROJECT_DIR1/logs"
LOG0="$LOG_DIR0/nccl_2node_rank0_${TS}.log"
LOG1="$LOG_DIR1/nccl_2node_rank1_${TS}.log"

BUILD0_CMD="cd '$PROJECT_DIR0' && nvcc -std=c++17 -O2 -o nccl_send_recv_2node nccl_send_recv_2node.cu -lnccl"
BUILD1_CMD="cd '$PROJECT_DIR1' && nvcc -std=c++17 -O2 -o nccl_send_recv_2node nccl_send_recv_2node.cu -lnccl"

RUN0_CMD="cd '$PROJECT_DIR0' && mkdir -p logs && \
  NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME='${IFACE}' \
  ./nccl_send_recv_2node --rank 0 --local-gpu ${GPU0} --master-addr '${NODE0_IP}' --master-port ${PORT} \
  --size-mb ${SIZE_MB} --warmup ${WARMUP} --iters ${ITERS} | tee '${LOG0}'"

RUN1_CMD="cd '$PROJECT_DIR1' && mkdir -p logs && \
  NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME='${IFACE}' \
  ./nccl_send_recv_2node --rank 1 --local-gpu ${GPU1} --master-addr '${NODE0_IP}' --master-port ${PORT} \
  --size-mb ${SIZE_MB} --warmup ${WARMUP} --iters ${ITERS} | tee '${LOG1}'"

echo "== Building on node0: ${NODE0} =="
ssh "$NODE0" "$BUILD0_CMD"

echo "== Building on node1: ${NODE1} =="
ssh "$NODE1" "$BUILD1_CMD"

echo "== Launching rank 0 on ${NODE0} =="
ssh "$NODE0" "$RUN0_CMD" &
PID0=$!

# Small delay so rank0 starts listener before rank1 tries to connect.
sleep 1

echo "== Launching rank 1 on ${NODE1} =="
ssh "$NODE1" "$RUN1_CMD" &
PID1=$!

set +e
wait "$PID0"
RC0=$?
wait "$PID1"
RC1=$?
set -e

echo ""
echo "Rank0 exit code: ${RC0}"
echo "Rank1 exit code: ${RC1}"
echo "Logs:"
echo "  ${NODE0}:${LOG0}"
echo "  ${NODE1}:${LOG1}"

if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  exit 1
fi

echo "2-node NCCL send/recv test completed."
