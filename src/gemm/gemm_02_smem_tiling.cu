#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"

// [补] TILE 原来没有定义。先固定为 16，之后做 16/32 对比时再改成模板
constexpr int TILE = 16;

// Example: M=N=K=4, TILE=2. C is cut into 2x2 tiles, one block per tile.
// blockIdx = (x=1, y=0) -> block computes C rows 0-1, cols 2-3 (upper-right tile).
// threadIdx = (x=0, y=1) -> thread computes C[1][2] (row = 0*2+1, col = 1*2+0).
// Needs A rows 0-1 (all K) and B cols 2-3 (all K), split into K/TILE rounds.
__global__ void gemm_smem_tiling_kernel(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C,
                                        int M, int N, int K)
{
  // [补] shared memory 声明
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // step 1: which C element this thread owns (x -> col, y -> row)
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.0f; // [补] 累加器原来没有声明

  // [补] 外层沿 K 方向的循环，step 2~4 都要放在这个循环里面
  const int numTiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < numTiles; ++t)
  {
    // step 2: A tile -> As[ty][tx] = A[row][t*TILE + tx]
    const int aCol = t * TILE + tx;
    As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

    // step 3: B tile -> Bs[ty][tx] = B[t*TILE + ty][col]
    const int bRow = t * TILE + ty;
    Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

    // step 4: barrier 1 (RAW) -> compute -> barrier 2 (WAR)
    __syncthreads(); // [改] 原来写成 _syncthreads，少了一个下划线
#pragma unroll
    for (int k = 0; k < TILE; ++k)
    {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  // step 5: write back
  if (row < M && col < N)
  {
    C[row * N + col] = acc;
  }
}

void gemm_02_smem_tiling(const float *A, const float *B, float *C,
                         int M, int N, int K)
{
  dim3 blockDim(TILE, TILE);
  dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_smem_tiling_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}