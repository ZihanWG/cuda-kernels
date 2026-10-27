#pragma once
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err_));                                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__,  \
              (int)st_);                                                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }
