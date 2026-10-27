#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"
#include <cstdio>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_vectorized_kernel(const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       float *__restrict__ C,
                                       int M, int N, int K)
{
    __shared__ __align__(16) float As[BK][BM]; // 转置
    __shared__ __align__(16) float Bs[BK][BN];

    const int cRow = blockIdx.y, cCol = blockIdx.x;
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // float4 搬运坐标
    const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);

    float acc[TM][TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK)
    {
        // TODO 1: 从 A 读 1 个 float4，转置后写进 As（4 行赋值）
        float4 tmp = reinterpret_cast<const float4 *>(
            &A[(cRow * BM + innerRowA) * K + (bk + innerColA * 4)])[0];
        As[innerColA * 4 + 0][innerRowA] = tmp.x;
        As[innerColA * 4 + 1][innerRowA] = tmp.y;
        As[innerColA * 4 + 2][innerRowA] = tmp.z;
        As[innerColA * 4 + 3][innerRowA] = tmp.w;
        // TODO 2: 从 B 读 1 个 float4，直接写进 Bs
        reinterpret_cast<float4 *>(&Bs[innerRowB][innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(
                &B[(bk + innerRowB) * N + (cCol * BN + innerColB * 4)])[0];
        __syncthreads();

#pragma unroll
        for (int dot = 0; dot < BK; ++dot)
        {
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                regM[i] = As[dot][threadRow * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                regN[j] = Bs[dot][threadCol * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    // TODO 4: float4 写回 C
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; j += 4)
        {
            reinterpret_cast<float4 *>(
                &C[(cRow * BM + threadRow * TM + i) * N + (cCol * BN + threadCol * TN + j)])[0] =
                make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
        }
    }
}
void gemm_05_vectorized(const float *A, const float *B, float *C,
                        int M, int N, int K, cudaStream_t stream)
{
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        fprintf(stderr, "gemm_05: M, N must be multiples of %d, K of %d\n", BM, BK);
        return;
    }
    dim3 block((BM / TM) * (BN / TN)); // 256
    dim3 grid(N / BN, M / BM);
    gemm_vectorized_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}