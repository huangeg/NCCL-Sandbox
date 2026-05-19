# NCCL-Sandbox Makefile
# Targets: nccl_send_recv_2node, nccl_rdma_bench
# GPU target: NVIDIA L4 (Ada Lovelace, sm_89)

CUDA_HOME := /opt/pytorch/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

INCLUDES  := -I$(CUDA_HOME)/include
LDFLAGS   := -L$(CUDA_HOME)/lib
# libnccl.so is versioned-only on this system; link by filename.
LIBS      := -l:libnccl.so.2 -lcudart

NVCCFLAGS := -std=c++17 -O2 \
             -gencode arch=compute_89,code=sm_89 \
             $(INCLUDES) \
             -Xcompiler "-Wall -O2"

.PHONY: all clean

all: nccl_send_recv_2node nccl_rdma_bench

nccl_send_recv_2node: nccl_send_recv_2node.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

nccl_rdma_bench: nccl_rdma_bench.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f nccl_send_recv_2node nccl_rdma_bench
