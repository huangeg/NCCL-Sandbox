# NCCL-Sandbox

A sandbox project for testing and benchmarking [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl) functionalities and performance.

## Purpose

NCCL-Sandbox provides a controlled environment to:

- **Explore NCCL primitives** – experiment with collective operations such as `AllReduce`, `AllGather`, `Broadcast`, `Reduce`, and `ReduceScatter`.
- **Benchmark performance** – measure throughput and latency of NCCL collectives across different hardware configurations and GPU topologies.
- **Validate correctness** – verify that collective results match expected outputs for various data types and reduction operations.
- **Prototype new ideas** – rapidly iterate on custom communication patterns and tuning strategies before integrating them into production workloads.

## Background

[NCCL](https://github.com/NVIDIA/nccl) is NVIDIA's high-performance library for multi-GPU and multi-node collective communication. It is widely used in distributed deep-learning frameworks (e.g., PyTorch, TensorFlow) to synchronise gradients and exchange activations efficiently across GPUs connected by NVLink, PCIe, or InfiniBand.

## Getting Started

### Prerequisites

- NVIDIA GPU(s) with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [NCCL](https://developer.nvidia.com/nccl/nccl-download) (version compatible with your CUDA installation)

### Building

```bash
# Clone the repository
git clone https://github.com/huangeg/NCCL-Sandbox.git
cd NCCL-Sandbox

# Build (adjust the Makefile or CMake flags as needed)
make
```

### Running

```bash
# Example: run a basic AllReduce test
./allreduce_test
```

Refer to individual source files and any in-directory `README` files for details on each test or benchmark.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

## License

See [LICENSE](LICENSE) for details.
