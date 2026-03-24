
## RTX5070Ti features/key info:
```
Number of SM = 70
Number of CUDA cores = 8960
Number of CUDA cores per SM = 128  
Max threads per SM (resident) = 2048
Max blocks per SM = 32  
Max number of threads per block = 1024  
Warp size = 32  
Peak BW = 896 GB/s  
Peak compute = 44 TFLOPS/s
```
## Formulas  
- Arithmetic Intensity (AI) = FLOPs / Bytes (for example: vector add -> 1 FLOPS, 12 bytes accessed = 1/12 = 0.083)  
- Roofline Performance = min(Compute Peak, Memory Bandwidth × Arithmetic Intensity). 
    - For memory intensive operation, peak BW and AI is limiting. 
    - For compute heavy, peak compute is limiting.
- FLOPS = numThreads * FLOPS per iteration * iterations / kernel_time (secs) * 10e-9  


## CUDA execution model
### Hierarchy: 
- Grid 
    - Blocks
        - Threads

### Built-in: 
```blockDim.x   // threads per block```  
```gridDim.x    // blocks per grid```  
```threadIdx.x  // thread id within block```  
```blockIdx.x   // block id within grid```  

### Global thread index
```idx = (blockIdx.x × blockDim.x) + threadIdx.x```   

### Total threads launched
```TotalThreads = gridDim.x × blockDim.x```   

### Thread scheduling in SM
- CUDA threads are grouped into warps of 32 threads.
- Each warp executes SIMD style on the cores.
- SM executes 1 warp at a time per instruction per cycle, or multiple warps in a pipeline if resources allow.

### Grid-Stride loop
#### Why to use
- Grid size independent of N
- Scales to very large problems (e.g., N = 1e9)
- Improves occupancy tuning flexibility
- Reuses threads efficiently
- Avoids launching extremely large grids

```
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = idx; i < N; i += stride)
    work(i); 
```

## __syncthreads -> stop execution, and wait until all threads in the block reach same point. 
### Example: 
```
__global__ void example()
{
    __shared__ int s[4];
    int tid = threadIdx.x;

    s[tid] = tid;     // Step 1: write

    __syncthreads();  // Without barrier, thread id 0 might read uninitialized value of s[tid+1] below. 

    int val = s[(tid + 1) % 4];   // Step 2: safe read
    printf("Thread %d sees %d\n", tid, val);
}
```

## Kernel Study Progression
### Memory-Bound Kernels:
- Vector add
- AXPY
- Reduction
- Scan
- Histogram

### Mixed Kernels:
- Convolution
- FFT
- Sparse Matrix-Vector Multiply

### Compute-Bound Kernels:
- Tiled GEMM
- FMA-heavy loops
- Tensor core GEMM

### Irregular Workloads:
- BFS
- PageRank
- N-body simulation

### AI-Oriented Kernels:
- Softmax
- Attention
- LayerNorm

#### Kernel Analysis Checklist

- Arithmetic intensity?
- Memory-bound or compute-bound?
- Occupancy level?
- Coalesced memory access?
- Shared memory bank conflicts?
- Warp divergence?
- Synchronization overhead?
- Launch configuration optimal?

    