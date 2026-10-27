#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"
#include <cstdio>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_2d_blocktiling_kernel(const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float *__restrict__ C,
                                           int M, int N, int K)
{
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN); // 256

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // 坐标 1：计算用
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // 坐标 2、3：搬运用
    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
    constexpr int strideA = NUM_THREADS / BK; // 256/bk = 32
    constexpr int strideB = NUM_THREADS / BN; // 256/bn = 2

    float acc[TM][TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < K; bk += BK)
    {
        // TODO 1: 循环搬 A，offset 从 0 到 BM，步长 strideA
        //   As[innerRowA + offset][innerColA] = A[ 全局行 ][ 全局列 ]
        for (int offset = 0; offset < BM; offset += strideA)
        {
            As[innerRowA + offset][innerColA] = A[(cRow * BM + innerRowA + offset) * K + (bk + innerColA)];
        }
        // TODO 2: 循环搬 B，offset 从 0 到 BK，步长 strideB
        //   Bs[innerRowB + offset][innerColB] = B[ 全局行 ][ 全局列 ]
        for (int offset = 0; offset < BK; offset += strideB)
        {
            Bs[innerRowB + offset][innerColB] = B[(bk + innerRowB + offset) * N + (cCol * BN + innerColB)];
        }
        __syncthreads();

        for (int dot = 0; dot < BK; ++dot)
        {
            // TODO 3: 读 8 个 A 到 regM
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                regM[i] = As[threadRow * TM + i][dot];
            }
#pragma unroll
            // TODO 4: 读 8 个 B 到 regN
            for (int j = 0; j < TN; ++j)
            {
                regN[j] = Bs[dot][threadCol * TN + j];
            }

            // TODO 5: 外积，累加到 acc[i][j]
            for (int i = 0; i < TM; ++i)
            {
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    // TODO 6: 把 64 个结果写回 C
    //   acc[i][j] 的全局行 = cRow*BM + threadRow*TM + i
    //   全局列 = ?
    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; ++j)
        {
            C[(cRow * BM + threadRow * TM + i) * N + (cCol * BN + threadCol * TN + j)] =
                acc[i][j];
        }
    }
}

void gemm_04_2d_blocktiling(const float *A, const float *B, float *C,
                            int M, int N, int K, cudaStream_t stream)
{
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        fprintf(stderr, "gemm_04: M, N must be multiples of %d, K of %d\n", BM, BK);
        return;
    }
    dim3 block((BM / TM) * (BN / TN)); // 256
    dim3 grid(N / BN, M / BM);
    gemm_2d_blocktiling_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}