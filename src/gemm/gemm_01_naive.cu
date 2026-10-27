#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"

// v1: one thread per C element. threadIdx.x walks the column, so a warp reads
// 32 consecutive floats of B and writes 32 consecutive floats of C (coalesced);
// A is a broadcast within the warp. Every operand comes straight from global
// memory, K times per output -- arithmetic intensity is ~0.25 FLOP/byte.
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

void gemm_01_naive(const float* A, const float* B, float* C,
                   int M, int N, int K, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
  gemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
