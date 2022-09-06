#include <iostream>

#include "test.hpp"

namespace
{
  __global__ void my_kernel(float *const in, int N)
  {
    auto const i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
    {
      return;
    }

    in[i] = 1.0f;
  }
}

void runKernelFromAnotherLib(float *ptr, int N)
{
  std::cout << "Hello from the CUDA file of another Library" << std::endl;
  my_kernel<<<1, N>>>(ptr, N);
}
