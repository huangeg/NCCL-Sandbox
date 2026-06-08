#!/usr/bin/env bash
# run_nccl_mpi_2node.sh
# Launch a 2-rank cross-node NCCL send/recv benchmark via MPI.
# Designed for AWS g7e.8xlarge with EFA + GPUDirect RDMA (aws-ofi-nccl).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_nccl_mpi_2node.sh [options]

Network options (required):
  --node0 <user@host>      SSH target for rank 0 (e.g. ec2-user@13.218.50.236)
  --node1 <user@host>      SSH target for rank 1 (e.g. ec2-user@3.83.165.50)
  --node0-ip <ip>          IP reachable from node1 for MPI/NCCL (e.g. 13.218.50.236)
  --node1-ip <ip>          IP reachable from node0 for MPI/NCCL (e.g. 3.83.165.50)

Path options (pick one):
  --project-dir <path>     Absolute path on BOTH nodes
  --project-dir0 <path>    Absolute path on node0 only
  --project-dir1 <path>    Absolute path on node1 only

Benchmark options:
  --gpu0 <id>              GPU index on node0 (default: 0)
  --gpu1 <id>              GPU index on node1 (default: 0)
  --size-mb <int>          Payload MiB per direction (default: 64)
  --warmup <int>           Warmup iterations (default: 5)
  --iters <int>            Timed iterations (default: 20)
  --iface <name>           NIC interface for NCCL socket bootstrap (default: eth0)
  --ssh-key <path>         SSH private key for node1 (default: ~/.ssh/id_rsa)
  --nccl-debug <level>     NCCL_DEBUG level: WARN, INFO, TRACE (default: WARN)
  --no-build               Skip build step (binary must already exist on both nodes)

Notes:
  - node0 must be reachable via mpirun (usually this local machine).
  - node1 requires passwordless SSH from node0.
  - Binaries must exist at the same path on both nodes; this script builds on
    each node via SSH then runs mpirun from node0.
  - aws-ofi-nccl plugin is loaded via LD_LIBRARY_PATH (no LD_PRELOAD needed).
  - FI_EFA_USE_DEVICE_RDMA=1 enables GPUDirect RDMA over EFA.
EOF
}

# ── Defaults ──────────────────────────────────────────────────────────────────
NODE0=""
NODE1=""
NODE0_IP=""
NODE1_IP=""
PROJECT_DIR=""
PROJECT_DIR0=""
PROJECT_DIR1=""
GPU0=0
GPU1=0
SIZE_MB=64
WARMUP=5
ITERS=20
IFACE="eth0"
SSH_KEY="${HOME}/.ssh/id_rsa"
NCCL_DEBUG_LEVEL="WARN"
NO_BUILD=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node0)          NODE0="$2";          shift 2 ;;
    --node1)          NODE1="$2";          shift 2 ;;
    --node0-ip)       NODE0_IP="$2";       shift 2 ;;
    --node1-ip)       NODE1_IP="$2";       shift 2 ;;
    --project-dir)    PROJECT_DIR="$2";    shift 2 ;;
    --project-dir0)   PROJECT_DIR0="$2";   shift 2 ;;
    --project-dir1)   PROJECT_DIR1="$2";   shift 2 ;;
    --gpu0)           GPU0="$2";           shift 2 ;;
    --gpu1)           GPU1="$2";           shift 2 ;;
    --size-mb)        SIZE_MB="$2";        shift 2 ;;
    --warmup)         WARMUP="$2";         shift 2 ;;
    --iters)          ITERS="$2";          shift 2 ;;
    --iface)          IFACE="$2";          shift 2 ;;
    --ssh-key)        SSH_KEY="$2";        shift 2 ;;
    --nccl-debug)     NCCL_DEBUG_LEVEL="$2"; shift 2 ;;
    --no-build)       NO_BUILD=true;       shift   ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# ── Validate required args ────────────────────────────────────────────────────
if [[ -z "$NODE0" || -z "$NODE1" || -z "$NODE0_IP" || -z "$NODE1_IP" ]]; then
  echo "ERROR: --node0, --node1, --node0-ip, and --node1-ip are required." >&2
  usage; exit 1
fi

if [[ -n "$PROJECT_DIR" && (-n "$PROJECT_DIR0" || -n "$PROJECT_DIR1") ]]; then
  echo "ERROR: use --project-dir OR (--project-dir0 and --project-dir1), not both." >&2
  exit 1
fi
[[ -n "$PROJECT_DIR" ]] && { PROJECT_DIR0="$PROJECT_DIR"; PROJECT_DIR1="$PROJECT_DIR"; }
if [[ -z "$PROJECT_DIR0" || -z "$PROJECT_DIR1" ]]; then
  echo "ERROR: provide --project-dir or both --project-dir0 and --project-dir1." >&2
  usage; exit 1
fi

MPIRUN=/opt/amazon/openmpi/bin/mpirun

# ── Environment variables forwarded to both ranks ────────────────────────────
# EFA + aws-ofi-nccl: loaded via LD_LIBRARY_PATH so NCCL auto-discovers the plugin.
OFI_NCCL_LIB=/opt/amazon/ofi-nccl/lib64
EFA_LIB=/opt/amazon/efa/lib64
CUDA_LIB=/opt/pytorch/cuda/lib

EFA_ENV=(
  "LD_LIBRARY_PATH=${OFI_NCCL_LIB}:${EFA_LIB}:${CUDA_LIB}:\${LD_LIBRARY_PATH:-}"
  "FI_EFA_USE_DEVICE_RDMA=1"       # enable GPUDirect RDMA over EFA
  "FI_EFA_FORK_SAFE=1"             # required when MPI forks helper processes
  "RDMAV_FORK_SAFE=1"              # ibverbs fork safety
  "NCCL_DEBUG=${NCCL_DEBUG_LEVEL}"
  "NCCL_SOCKET_IFNAME=${IFACE}"    # bootstrap TCP interface
  "NCCL_IB_DISABLE=0"              # allow EFA (EFA appears as IB to NCCL)
)

# Build -x flags for mpirun
MPI_X_FLAGS=()
for env_var in "${EFA_ENV[@]}"; do
  MPI_X_FLAGS+=(-x "$env_var")
done

# ── Build step ────────────────────────────────────────────────────────────────
ssh_cmd() {
  local target="$1"; shift
  ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
      -i "$SSH_KEY" "$target" "$@"
}

BUILD_CMD="cd '{}' && make nccl_mpi_2node 2>&1 | tail -5"

if [[ "$NO_BUILD" == false ]]; then
  echo "── Building on node0 (${NODE0}) ──"
  node0_build_cmd="${BUILD_CMD/\{\}/$PROJECT_DIR0}"
  if [[ "${NODE0#*@}" == "$(hostname)" || "${NODE0#*@}" == "$(hostname -s)" || \
        "${NODE0#*@}" == "localhost" || "${NODE0#*@}" == "127.0.0.1" ]]; then
    bash -c "$node0_build_cmd"
  else
    ssh_cmd "$NODE0" "bash -c \"${node0_build_cmd}\""
  fi

  echo "── Building on node1 (${NODE1}) ──"
  node1_build_cmd="${BUILD_CMD/\{\}/$PROJECT_DIR1}"
  ssh_cmd "$NODE1" "bash -c \"${node1_build_cmd}\""
fi

# ── Hostfile ──────────────────────────────────────────────────────────────────
HOSTFILE=$(mktemp /tmp/nccl_mpi_hosts.XXXXXX)
trap 'rm -f "$HOSTFILE"' EXIT
cat > "$HOSTFILE" <<EOF
${NODE0_IP} slots=1
${NODE1_IP} slots=1
EOF

echo ""
echo "── Hostfile ──"
cat "$HOSTFILE"
echo ""

# ── Log setup ────────────────────────────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR0}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/nccl_mpi_2node_${TS}.log"

# ── Launch ────────────────────────────────────────────────────────────────────
echo "── Launching 2-rank NCCL MPI benchmark ──"
echo "  node0: ${NODE0_IP} (GPU ${GPU0})"
echo "  node1: ${NODE1_IP} (GPU ${GPU1})"
echo "  size : ${SIZE_MB} MiB  warmup: ${WARMUP}  iters: ${ITERS}"
echo "  log  : ${LOG_FILE}"
echo ""

set -x
"$MPIRUN" \
  -np 2 \
  --hostfile "$HOSTFILE" \
  -mca pml ob1 \
  -mca btl self,vader \
  -mca mtl ofi \
  -mca opal_common_ofi_provider_include efa \
  "${MPI_X_FLAGS[@]}" \
  --bind-to none \
  --rank-by node \
  -wdir "$PROJECT_DIR0" \
  "${PROJECT_DIR0}/nccl_mpi_2node" \
    --local-gpu 0 \
    --size-mb "${SIZE_MB}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
  2>&1 | tee "$LOG_FILE"
RC=${PIPESTATUS[0]}
set +x

echo ""
if [[ $RC -eq 0 ]]; then
  echo "Done. Log: ${LOG_FILE}"
else
  echo "ERROR: mpirun exited with code ${RC}. See ${LOG_FILE}" >&2
  exit $RC
fi
