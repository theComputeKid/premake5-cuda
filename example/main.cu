#include <cuda_runtime.h>

__global__ void cuda_hello();

int runKernel() {
    cuda_hello << <1, 1 >> > ();
    cudaDeviceSynchronize();
    return 0;
}
