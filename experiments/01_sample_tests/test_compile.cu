#include <stdio.h>
#include <unistd.h> // for sleep

__global__ void hello() {
    printf("Hello from GPU!\n");
}

int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    sleep(10); // keeps process alive for 10 seconds
    return 0;
}