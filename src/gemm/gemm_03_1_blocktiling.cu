#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"
#include <cstdio>

template <int BM, int BN, int BK, int TM>
__global__ void gemm_1d_blocktiling_kernel(const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float *__restrict__ C,
                                           int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int cRow = blockIdx.y; // 第几个 BM 行块
    const int cCol = blockIdx.x; // 第几个 BN 列块

    // 坐标 1：计算用
    const int threadCol = threadIdx.x % BN; // 第几个 TM 列
    const int threadRow = threadIdx.x / BN; // 第几个 TM 行
    // 坐标 2、3：搬运用
    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;

    float acc[TM] = {0.0f}; // 8 个小本子，编译器会放进寄存器

    for (int bk = 0; bk < K; bk += BK)
    {
        // TODO 1: As[innerRowA][innerColA] = A[ 全局行 ][ 全局列 ]
        //         全局行 = cRow*BM + innerRowA，全局列 = bk + innerColA
        As[innerRowA][innerColA] = A[(cRow * BM + innerRowA) * K + (bk + innerColA)];
        // TODO 2: Bs[innerRowB][innerColB] = B[ 全局行 ][ 全局列 ]
        //         全局行 = bk + innerRowB，全局列 = cCol * BN + innerColB
        Bs[innerRowB][innerColB] = B[(bk + innerRowB) * N + (cCol * BN + innerColB)];

        __syncthreads();

        for (int dot = 0; dot < BK; ++dot)
        {
            // TODO 3: 把 Bs 的一个值读进寄存器 b
            float b = Bs[dot][threadCol];

            for (int i = 0; i < TM; ++i)
            {
                // TODO 4: acc[i] += ...
                acc[i] += As[threadRow * TM + i][dot] * b;
            }
        }
        __syncthreads();
    }

    // TODO 5: 把 8 个结果写回 C
    //         第 i 个结果的全局行 = cRow*BM + threadRow*TM + i，全局列 = ?
    for (int i = 0; i < TM; ++i)
    {
        C[(cRow * BM + threadRow * TM + i) * N + (cCol * BN + threadCol)] = acc[i];
    }
}



void gemm_03_1d_blocktiling(const float *A, const float *B, float *C,
                            int M, int N, int K, cudaStream_t stream)
{
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    // assert() is compiled out in Release builds, so check at runtime instead
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        fprintf(stderr,
                "gemm_03_1d_blocktiling: M=%d N=%d K=%d must be multiples of "
                "BM=%d BN=%d BK=%d; skipping launch\n",
                M, N, K, BM, BN, BK);
        return;
    }
    dim3 block(BM * BN / TM); // 512 个线程，一维
    dim3 grid(N / BN, M / BM);
    gemm_1d_blocktiling_kernel<BM, BN, BK, TM>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}