/**
 * @file main.cu
 *
 * @brief main CUDA file of the project. To be compiled into PTX.
 *
 * @version 0.1
 * @date 2022-09-03
 */
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

namespace
{
  std::size_t constexpr N = 10;

  __global__ void my_kernel(float *const in)
  {
    auto const i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
    {
      return;
    }

    in[i] = 1.0f;
  }
}

void cuda_kernel_ptx()
{
  std::cout << "Hello from the PTX file of the executable" << std::endl;
  std::vector<float> inHost(N, 1.0f);
  thrust::device_vector<float> inGPU(inHost);

  my_kernel<<<1, N>>>(thrust::raw_pointer_cast(inGPU.data()));
}
