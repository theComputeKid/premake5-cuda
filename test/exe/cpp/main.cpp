/**
 * @file main.cpp
 *
 * @brief Main entry point to CUDA executable project.
 *
 * @version 0.1
 * @date 2022-09-03
 */
#include <iostream>

/** @brief Imported from the .cu file. */
void cuda_kernel();

int main()
{
  std::cout << "Hello from the C++ file of the executable" << std::endl;
  cuda_kernel();
  return 0;
}
