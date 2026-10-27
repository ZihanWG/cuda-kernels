#pragma once
#include <cuda_runtime.h>

// All kernels compute C = A * B in FP32, row-major:
//   A: M x K, B: K x N, C: M x N  (no alpha/beta)
using GemmFn = void (*)(const float* A, const float* B, float* C,
                        int M, int N, int K, cudaStream_t stream);

void gemm_cublas(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream);
void gemm_01_naive(const float* A, const float* B, float* C,
                   int M, int N, int K, cudaStream_t stream);
void gemm_02_smem_tiling_t16(const float* A, const float* B, float* C,
                             int M, int N, int K, cudaStream_t stream);
void gemm_02_smem_tiling_t32(const float* A, const float* B, float* C,
                             int M, int N, int K, cudaStream_t stream);
void gemm_03_1d_blocktiling(const float* A, const float* B, float* C,
                            int M, int N, int K, cudaStream_t stream);
void gemm_04_2d_blocktiling(const float* A, const float* B, float* C,
                            int M, int N, int K, cudaStream_t stream);
void gemm_05_vectorized(const float* A, const float* B, float* C,
                        int M, int N, int K, cudaStream_t stream);
// Next versions: declare here, add a file in src/gemm/, then register in
// CMakeLists.txt and bench/bench_gemm.cu.
