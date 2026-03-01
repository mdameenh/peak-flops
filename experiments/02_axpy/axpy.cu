#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                   \
{                                                                     \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        printf("CUDA error at %s:%d -> %s\n",                         \
               __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Grid-stride vector addition
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
        C[i] = A[i] + B[i];
}

__global__
void axpy_kernel(float alpha,
                 const float* __restrict__ x,
                 float* __restrict__ y,
                 int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
        y[i] = alpha * x[i] + y[i];
}

int main()
{
    const float alpha = 2.5f;

    int blockSize = 256;
    int M = 50;  // Number of repetitions per N
    int startN = 0.1e9;
    int endN = 2.0e9;
    int stepN = 0.1e9;

    // Open CSV file for logging
    FILE* fp = fopen("experiments/02_axpy/data/axpy_results.csv", "w");
    if (!fp) {
        printf("Failed to open CSV file for writing.\n");
        return -1;
    }
    fprintf(fp, "N,run_index,kernel_time_ms,effective_bandwidth_GBps\n");

    // Loop over vector sizes
    for (int N = startN; N <= endN; N += stepN)
    {
        size_t bytes = N * sizeof(float);
        printf("----------------------------------------------------\n");
        printf("Vector size: %d elements\n", N);
        printf("Total data moved: %.2f MB\n", (3.0 * bytes) / (1024.0*1024.0));

        // Allocate host memory
        float *h_x = (float*)malloc(bytes);
        float *h_y = (float*)malloc(bytes);

        for (int i = 0; i < N; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
        }

        // Allocate device memory
        float *d_x, *d_y;
        CHECK(cudaMalloc(&d_x, bytes));
        CHECK(cudaMalloc(&d_y, bytes));

        // Copy to device
        CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

        // Determine grid size
        int maxSM;
        CHECK(cudaDeviceGetAttribute(&maxSM, cudaDevAttrMultiProcessorCount, 0));
        int gridSize = maxSM * 32;
        printf("Grid size: %d\n", gridSize);
        printf("Block size: %d\n", blockSize);

        // Prepare CUDA events
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        // Repeat M times
        for (int run = 1; run <= M; run++) {

            CHECK(cudaEventRecord(start));
            axpy_kernel<<<gridSize, blockSize>>>(alpha, d_x, d_y, N);
            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));

            float milliseconds = 0;
            CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

            // Copy result back (small check optional)
            CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

            int valid = 1;
            for (int i = 0; i < 10; i++) {
                float expected = alpha * 1.0f + 2.0f;
                if (fabs(h_y[i] - expected) > 1e-5) {
                    valid = 0;
                    break;
                }
                printf("expected %.8f, actual %.8f, valid %d\n", expected, h_y[i], valid);
            }

            double seconds = milliseconds / 1000.0;
            double totalBytes = 3.0 * bytes;
            double bandwidth = totalBytes / seconds / 1e9; // GB/s

            printf("Run %d: kernel time %.4f ms, bandwidth %.2f GB/s, verification: %s\n",
                   run, milliseconds, bandwidth, valid ? "PASSED" : "FAILED");

            // Log to CSV
            fprintf(fp, "%d,%d,%.4f,%.2f\n", N, run, milliseconds, bandwidth);
        }

        // Cleanup for this N
        free(h_x);
        free(h_y);
        CHECK(cudaFree(d_x));
        CHECK(cudaFree(d_y));

        printf("\n");
    }

    fclose(fp);
    printf("Benchmark complete. Results logged to axpy_results.csv\n");
    return 0;
}