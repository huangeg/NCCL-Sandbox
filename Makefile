# NCCL-Sandbox Makefile
# Targets: nccl_send_recv_2node, nccl_rdma_bench, nccl_mpi_2node
# GPU target: NVIDIA RTX PRO 6000 (Ada Lovelace, sm_89)

CUDA_HOME := /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

NCCL_HOME := /usr/local/cuda-13.0
MPI_HOME  := /opt/amazon/openmpi

INCLUDES  := -I$(CUDA_HOME)/include
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
# libnccl.so is versioned-only on this system; link by filename.
LIBS      := -l:libnccl.so.2 -lcudart

MPI_INC     := -I$(MPI_HOME)/include
MPI_LDFLAGS := -L$(MPI_HOME)/lib64 -Xlinker -rpath -Xlinker $(MPI_HOME)/lib64
MPI_LIBS    := -lmpi

NVCCFLAGS := -std=c++17 -O2 \
             -gencode arch=compute_89,code=sm_89 \
             $(INCLUDES) \
             -Xcompiler "-Wall -O2"

.PHONY: all clean

all: nccl_send_recv_2node nccl_rdma_bench nccl_mpi_2node

nccl_send_recv_2node: nccl_send_recv_2node.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

nccl_rdma_bench: nccl_rdma_bench.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

nccl_mpi_2node: nccl_mpi_2node.cu
	$(NVCC) $(NVCCFLAGS) $(MPI_INC) $(LDFLAGS) $(MPI_LDFLAGS) -o $@ $< $(LIBS) $(MPI_LIBS)

clean:
	rm -f nccl_send_recv_2node nccl_rdma_bench nccl_mpi_2node
