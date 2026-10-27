# CUDA GEMM from Scratch (cuda-kernels)

Hand-written SGEMM kernels optimized step by step, each version compared against cuBLAS and profiled with Nsight Compute. The goal is not just speed, but being able to explain *why* each step is faster and where the bottleneck moves next.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3080 Ti Laptop (Ampere, sm_86, 58 SMs, 16 GB, 175 W) |
| OS | Windows 11 + WSL2 Ubuntu 24.04 |
| Toolkit | CUDA 13.3, Nsight Compute |
| Precision | FP32, row-major, C = A · B (TF32 disabled in cuBLAS for a fair comparison) |

## Methodology

- **Correctness**: every kernel is checked against cuBLAS (max-abs error normalized, threshold 1e-3). Output buffers are pre-filled with NaN so unwritten elements are caught.
- **Timing**: warm-up runs, then the median of multiple timed runs. cuBLAS and all kernels run back-to-back in each round so they see the same thermal state.
- **Primary metric**: % of cuBLAS rather than absolute GFLOPS, because a laptop GPU throttles with power and temperature.
- **Profiling**: Nsight Compute at 4096³, one pass per kernel.

## Results

| Size | cuBLAS | 01 naive | 02 smem tiling | 02 vs 01 |
|---|---|---|---|---|
| 512 | 6.1 TFLOPS | 1092 GFLOPS (17.9%) | 1524 GFLOPS (25.0%) | 1.40× |
| 1024 | 10.8 TFLOPS | 1206 GFLOPS (11.2%) | 1642 GFLOPS (15.2%) | 1.36× |
| 2048 | 14.5 TFLOPS | 1429 GFLOPS (9.9%) | 2032 GFLOPS (14.1%) | 1.42× |
| 4096 | 15.7 TFLOPS | 1341 GFLOPS (8.6%) | 1937 GFLOPS (12.4%) | 1.44× |

All kernels PASS correctness at every size.

## Kernels

### 01 — naive

One thread computes one element of C, reading a full row of A and a full column of B directly from global memory. `threadIdx.x` maps to the column, so loads of B and stores of C are coalesced. Block size 32×32.

### 02 — shared memory tiling (TILE = 16)

Each block computes a 16×16 tile of C. The K dimension is split into K/TILE rounds; in each round the block cooperatively loads a 16×16 tile of A and of B into shared memory, then each thread accumulates 16 FMAs from shared memory.

- **Two barriers per round**: the first prevents reading tiles before every thread has written them (RAW); the second prevents a fast thread from overwriting tiles that others are still reading (WAR).
- **Bounds handling**: out-of-range threads write 0 into shared memory instead of returning early, because every thread in the block must reach `__syncthreads()`.
- **Coalescing**: for both A and B loads, `threadIdx.x` walks contiguous addresses.
- **Bank conflicts**: reading `As[ty][k]` is a broadcast, reading `Bs[k][tx]` hits 32 distinct banks, so the standard layout is conflict-free.

In theory this cuts global-memory reads per thread from 2K to 2K/TILE.

## Profiling: why only 1.4×?

Nsight Compute, 4096³:

| Metric | 01 naive (32×32) | 02 smem (16×16) |
|---|---|---|
| Elapsed cycles | 160.9 M | 113.8 M (1.41× fewer) |
| L1/TEX throughput | 93.9% | 97.8% |
| DRAM throughput | 16.1% | 41.3% |
| DRAM bytes read | 9.12 GB | 17.42 GB |
| L2 throughput | 11.7% | 31.3% |
| L1 hit rate | 95.0% | 0.13% |
| Achieved occupancy | 66.7% | 99.8% |
| Issue slots busy | 26.8% | 31.0% |
| IPC (active) | 1.09 | 1.24 |
| Smem load bank conflicts | — | 248,751 / 3.22 G wavefronts |

### Findings

1. **Both kernels are LSU-bound, not DRAM- or compute-bound.** L1/LSU throughput is saturated (94–98%) while DRAM sits at 16–41% and issue slots are only ~30% busy, so the FMA units are mostly idle. In 02, every 16 FMAs require 32 shared-memory loads.
2. **The naive kernel was already being tiled by hardware.** Its 32×32 block gives a 95% L1 hit rate, which acts as an implicit TILE=32. The explicit TILE=16 in 02 actually has *less* reuse, which explains the modest speedup and why DRAM bytes read doubled (02's global loads miss L1 almost entirely).
3. **Occupancy matches the theory.** A 1024-thread block fits only once into an SM's 1536-thread limit (2/3 occupancy); 256-thread blocks reach ~100%.
4. **Bank conflicts are negligible** for loads (~0.01%). Store conflicts exist but are small.

**Takeaway**: shared-memory tiling moved the pressure from global memory onto the LSU pipeline. The next step must raise the number of FMAs per load instruction, which is what register tiling does.

## Roadmap

- [x] 01 naive baseline + benchmark harness
- [x] 02 shared memory tiling + Nsight Compute analysis
- [ ] TILE=32 experiment: higher reuse vs lower occupancy
- [ ] 03 register tiling (each thread computes a 4×4 / 8×8 block of C)
- [ ] 04 vectorized loads (float4), 05 double buffering
- [ ] Phase 2: FlashAttention-style fused attention (CUDA + Triton), vs PyTorch SDPA / FlashAttention-2
- [ ] Phase 3: inference systems with vLLM (PagedAttention, KV cache, batching, quantization)

## Build & run
    cmake -S . -B build && cmake --build build -j
    ./build/bench_gemm                       # all kernels, default sizes
    ./build/bench_gemm --sizes 1024,2048 --kernels naive --iters 30

## Layout
- `src/gemm/gemm.h` – common signature; one file per kernel version (`gemm_NN_name.cu`)
- `src/gemm/cublas_ref.cu` – cuBLAS baseline (FP32, no TF32)
- `bench/bench_gemm.cu` – correctness + GFLOPS + % of cuBLAS (interleaved rounds, median)

## Adding a kernel
1. new `src/gemm/gemm_NN_xxx.cu`, declare in `gemm.h`
2. add to `gemm_kernels` in `CMakeLists.txt`
3. add to `kKernels` in `bench/bench_gemm.cu`
