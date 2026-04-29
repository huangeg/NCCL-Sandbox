# NCCL-Sandbox
Project to test NCCL functionalities and performance

## Run testing script
```
nvcc -std=c++17 -O2 -o nccl_perf_test nccl_perf_test.cu -lnccl -lpthread
./nccl_perf_test
```