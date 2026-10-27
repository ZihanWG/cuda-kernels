#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"

// cuBLAS is column-major. Row-major C = A*B is the same memory as
// column-major C^T = B^T * A^T, so swap the operands and dimensions.
void gemm_cublas(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  static cublasHandle_t handle = [] {
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    // Strict FP32 (no TF32 tensor cores) so the comparison is FP32 CUDA-core vs FP32.
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    return h;
  }();
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
}
