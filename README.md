# NCCL-Sandbox
Project to test NCCL functionalities and performance

## Run testing script
```
nvcc -std=c++17 -O2 -o nccl_perf_test nccl_perf_test.cu -lnccl -lpthread
./nccl_perf_test
```

## 2-node NCCL Send/Recv PoC (Ethernet/TCP)

This PoC focuses on point-to-point NCCL `send/recv` across two nodes and logs transfer speed.

### Files
- `nccl_send_recv_2node.cu`: rank-0/rank-1 benchmark program.
- `run_nccl_send_recv_2node.sh`: helper script to compile and launch both ranks over SSH.

### What this test does
- Runs 2 NCCL ranks (one process per node, one GPU per rank).
- Uses `ncclSend` + `ncclRecv` in each iteration (full-duplex exchange).
- Reports per-iteration and summary bandwidth in GB/s.

### Force NCCL to socket transport
The launcher sets:
- `NCCL_IB_DISABLE=1` (disable InfiniBand/RDMA path)
- `NCCL_SOCKET_IFNAME=<iface>` (select Ethernet NIC, for example `eth0`)

### Build manually
Run on each node:
```bash
cd /path/to/NCCL-Sandbox
nvcc -std=c++17 -O2 -o nccl_send_recv_2node nccl_send_recv_2node.cu -lnccl
```

### Run with helper script
Run this from your local machine (where SSH can reach both nodes):
```bash
cd /path/to/NCCL-Sandbox
chmod +x run_nccl_send_recv_2node.sh

./run_nccl_send_recv_2node.sh \
  --node0 gehuang@goofy-1 \
  --node1 gehuang@goofy-2 \
  --node0-ip 10.129.96.44 \
  --project-dir /home/gehuang/projects/NCCL-Sandbox \
  --gpu0 0 \
  --gpu1 0 \
  --size-mb 64 \
  --warmup 5 \
  --iters 20 \
  --port 50050 \
  --iface bond0
```

If the repository path is different on each node, use:
```bash
./run_nccl_send_recv_2node.sh \
	--node0 gehuang@goofy-1 \
	--node1 gehuang@goofy-2 \
	--node0-ip 10.129.96.44 \
	--project-dir0 /home/gehuang/projects/NCCL-Sandbox \
	--project-dir1 /mnt/work/NCCL-Sandbox \
	--gpu0 7 \
	--gpu1 7 \
	--size-mb 64 \
	--warmup 5 \
	--iters 20 \
	--port 50050 \
	--iface bond0
```

Logs are written to `logs/` under the project directory on both nodes.