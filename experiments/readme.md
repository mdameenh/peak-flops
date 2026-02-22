# Experiments for GPU & CUDA Study

This document lists practical experiments to deepen your understanding of GPU architecture, memory hierarchy, CUDA programming, kernels, streams, profiling, deep learning frameworks, optimization, and C++. Each experiment starts simple and builds up. Performance measurement and analysis notes are included.

---

## 1. GPU Architecture & Memory Hierarchy
- **Experiment:** Compare CPU vs. GPU performance for vector addition.
  - Implement vector addition on CPU and GPU.
  - Measure execution time and analyze speedup.
  - [CUDA Samples: vectorAdd](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAdd)
- **Experiment:** Test global vs. shared memory access in CUDA.
  - Write kernels using global and shared memory.
  - Measure latency and bandwidth.
  - [CUDA Memory Hierarchy Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)

## 2. CUDA Programming Basics
- **Experiment:** Write a basic CUDA kernel for element-wise array operations.
  - Launch kernels with different grid/block sizes.
  - Observe impact on performance.
  - [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## 3. CUDA Kernels, Streams, and Profiling
- **Experiment:** Use multiple CUDA streams for concurrent kernel execution.
  - Compare single vs. multiple streams.
  - [CUDA Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- **Experiment:** Profile a CUDA program using Nsight or Visual Profiler.
  - Identify bottlenecks and optimize kernel launches.
  - [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-visual-studio-edition)

## 4. Deep Learning Frameworks
- **Experiment:** Run a simple neural network on GPU using PyTorch or TensorFlow.
  - Compare CPU vs. GPU training time.
  - [PyTorch CUDA](https://pytorch.org/docs/stable/notes/cuda.html), [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- **Experiment:** Use cuDNN or TensorRT for inference acceleration.
  - Measure inference latency and throughput.
  - [cuDNN](https://developer.nvidia.com/cudnn), [TensorRT](https://developer.nvidia.com/tensorrt)

## 5. Optimization Guides
- **Experiment:** Optimize memory access patterns in CUDA kernels.
  - Use coalesced memory access and shared memory.
  - Profile before and after optimization.
  - [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## 6. C++ for CUDA
- **Experiment:** Implement a templated CUDA kernel for vector addition.
  - Use C++ templates for type flexibility.
  - [CUDA C++ Templates Example](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/template/template.cu)

---

## Performance Measurement & Analysis Notes
- Use `cudaEvent_t` for timing CUDA kernels ([CUDA Events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)).
- Use Nsight, Visual Profiler, or `nvprof` for profiling.
- Measure:
  - Execution time (wall-clock, kernel time)
  - Memory bandwidth (GB/s)
  - Occupancy and utilization
  - Latency and throughput
- Analyze:
  - Bottlenecks (memory, compute, launch overhead)
  - Impact of grid/block size
  - Effectiveness of optimization (before/after profiling)

---
