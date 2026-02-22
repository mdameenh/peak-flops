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

int main()
{
    int blockSize = 256;
    int M = 10;  // Number of repetitions per N
    int startN = 0.1e9;
    int endN = 2e9;
    int stepN = 0.1e9;

    // Open CSV file for logging
    FILE* fp = fopen("experiments/01_sample_tests/data/vecadd_results.csv", "w");
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
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);
        if (!h_A || !h_B || !h_C) {
            printf("Host allocation failed for N=%d\n", N);
            break;
        }

        // Initialize host arrays
        for (int i = 0; i < N; i++) {
            h_A[i] = 1.0f;
            h_B[i] = 2.0f;
        }

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc(&d_A, bytes));
        CHECK(cudaMalloc(&d_B, bytes));
        CHECK(cudaMalloc(&d_C, bytes));

        // Copy to device
        CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

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
            VecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));

            float milliseconds = 0;
            CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

            // Copy result back (small check optional)
            CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

            int valid = 1;
            for (int i = 0; i < 5; i++) {
                if (h_C[i] != 3.0f) {
                    valid = 0;
                    break;
                }
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
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);

        printf("\n");
    }

    fclose(fp);
    printf("Benchmark complete. Results logged to vecadd_results.csv\n");
    return 0;
}