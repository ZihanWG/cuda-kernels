#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"
#include <cstdio>

//04 让每个线程负责一个 8×8 的方块，A 和 B 的值都能复用 8 次：

//版本	每线程输出数     每轮 dot 读 smem	每轮 dot 做 FMA	   smem 读取 / FMA
//02	1	            2	              1	                2.0
//03	8（1 列）	    1 + 8 = 9	      8       	        约 1.1
//04	64（8×8）	    8 + 8 = 16	        64	            0.25

//厨房比喻： 每个厨师负责菜单上一个 8×8 的区域。每一步，左手拿 8 样 A 食材，右手拿 8 样 B 食材，两两组合做出 64 道菜。拿 16 次，做 64 次。

//数学上，这就是外积（outer product）： 一个 8 维列向量乘一个 8 维行向量，得到 8×8 的矩阵。
//
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
    
        //         全局行 = cRow*BM + innerRowA，全局列 = bk + innerColA
        As[innerRowA][innerColA] = A[(cRow * BM + innerRowA) * K + (bk + innerColA)];
        
        //         全局行 = bk + innerRowB，全局列 = cCol * BN + innerColB
        Bs[innerRowB][innerColB] = B[(bk + innerRowB) * N + (cCol * BN + innerColB)];

        __syncthreads();

        for (int dot = 0; dot < BK; ++dot)
        {
            
            float b = Bs[dot][threadCol];

            for (int i = 0; i < TM; ++i)
            {
                
                acc[i] += As[threadRow * TM + i][dot] * b;
            }
        }
        __syncthreads();
    }

  
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