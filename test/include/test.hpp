/**
 * @file test.hpp
 *
 * @brief Interface b/w lib and exe.
 *
 * @version 0.1
 * @date 2022-09-06
 */
#pragma once

#ifdef __cplusplus
#define PREMAKE_CUDA_EXTERN_C extern "C"
#endif

#ifdef _WIN32
#ifdef PREMAKE_CUDA_EXPORT_API
#define PREMAKE_CUDA_LINKAGE PREMAKE_CUDA_EXTERN_C __declspec(dllexport)
#else
#define PREMAKE_CUDA_LINKAGE PREMAKE_CUDA_EXTERN_C __declspec(dllimport)
#endif
#else
#define PREMAKE_CUDA_LINKAGE PREMAKE_CUDA_EXTERN_C __attribute__((visibility("default")))
#endif

PREMAKE_CUDA_LINKAGE void runKernelFromAnotherLib(float *ptr, int N);
