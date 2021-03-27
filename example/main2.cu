#include <stdio.h>
#include <cuda_runtime.h>

__global__ void cuda_hello() {
    printf("Hello from the GPU!\n");
}
