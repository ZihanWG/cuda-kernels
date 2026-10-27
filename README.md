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

| Size | cuBLAS | 01 naive | 02 T=16 | 02 T=32 | 02 T=16 vs 01 |
|---|---|---|---|---|---|
| 512 | 6.1 TFLOPS | 1088 GFLOPS (17.9%) | 1524 GFLOPS (25.0%) | 1394 GFLOPS (22.9%) | 1.40× |
| 1024 | 10.8 TFLOPS | 1208 GFLOPS (11.2%) | 1634 GFLOPS (15.1%) | 1500 GFLOPS (13.9%) | 1.35× |
| 2048 | 13.8 TFLOPS | 1309 GFLOPS (9.5%) | 1853 GFLOPS (13.5%) | 1678 GFLOPS (12.2%) | 1.42× |
| 4096 | 15.2 TFLOPS | 1253 GFLOPS (8.3%) | 1724 GFLOPS (11.4%) | 1653 GFLOPS (10.9%) | 1.38× |

All numbers are from one interleaved run. At 4096 the laptop GPU keeps throttling over consecutive runs, and in repeated runs the T=32 / T=16 time ratio ranged from 0.96× to 1.37×, so wall-clock ordering at 4096 is not reliable; the cycle counts from Nsight Compute below are the evidence used for conclusions.

All kernels PASS correctness at every size.

## Kernels

### 01 — naive

One thread computes one element of C, reading a full row of A and a full column of B directly from global memory. `threadIdx.x` maps to the column, so loads of B and stores of C are coalesced. Block size 32×32.

### 02 — shared memory tiling (TILE = 16 / 32, template parameter)

Each block computes a 16×16 tile of C. The K dimension is split into K/TILE rounds; in each round the block cooperatively loads a 16×16 tile of A and of B into shared memory, then each thread accumulates 16 FMAs from shared memory.

- **Two barriers per round**: the first prevents reading tiles before every thread has written them (RAW); the second prevents a fast thread from overwriting tiles that others are still reading (WAR).
- **Bounds handling**: out-of-range threads write 0 into shared memory instead of returning early, because every thread in the block must reach `__syncthreads()`.
- **Coalescing**: for both A and B loads, `threadIdx.x` walks contiguous addresses.
- **Bank conflicts**: reading `As[ty][k]` is a broadcast, reading `Bs[k][tx]` hits 32 distinct banks, so the standard layout is conflict-free.

In theory this cuts global-memory reads per thread from 2K to 2K/TILE. `TILE` is a template parameter; `02_smem_t16` and `02_smem_t32` are registered as separate benchmark entries. The profiling table below is for T=16.

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

## Experiment: TILE = 32

**Prediction.** A bigger tile should help on the memory side: global/L2 traffic per FMA halves, so L2 and DRAM pressure drops. The cost is occupancy: a 1024-thread block fits once per SM (1536-thread limit), so only 66.7% of the warp slots are used, and with a single resident block all warps wait at the same `__syncthreads()`. The roadmap framed it as "higher reuse vs lower occupancy"; my specific guess was that barrier stalls would be what makes T=32 lose.

**Nsight Compute, 4096³** (one pass each):

| Metric | T=16 | T=32 |
|---|---|---|
| Elapsed cycles | 115.4 M | 124.9 M (+8.3%) |
| DRAM bytes read | 17.58 GB | 8.99 GB |
| L2 throughput | 31.3% | 14.4% |
| L1/TEX (LSU) throughput | 97.7% | 82.3% |
| Achieved occupancy | 99.8% | 66.7% |
| Active / eligible warps per scheduler | 11.97 / 1.32 | 8.00 / 1.06 |
| Issue slots busy / IPC | 30.9% / 1.24 | 22.1% / 0.89 |
| Smem load wavefronts | 3.2216 G | 3.2212 G |
| Smem load / store bank conflicts | 255 K / 12.9 M | 0 / 34.6 M |

Warp stall breakdown (cycles per issued instruction; total 38.5 for T=16, 34.9 for T=32):

| Stall reason | T=16 | T=32 |
|---|---|---|
| MIO throttle | 18.9 (49%) | 21.1 (60%) |
| Long scoreboard (global load latency) | 7.0 (18%) | 4.8 (14%) |
| **Barrier** | **5.6 (15%)** | **2.6 (7%)** |
| Not selected / wait / other | ~7 | ~7 |

**Result.** T=32 does what the prediction said on the memory side: DRAM reads halve, L2 throughput halves, LSU load drops from 98% to 82%. It is still ~8% slower in cycles, and benchmarks show T=16 ahead by 8–12% up to 2048 (4096 is noisy, see above).

**The barrier guess was wrong.** Barrier stalls are *smaller* with T=32 (7% vs 15% of stall cycles); a larger block spends fewer barrier rounds per useful FMA. The loss comes from latency hiding instead: with 8 instead of 12 warps per scheduler there are fewer eligible warps (1.06 vs 1.32), IPC falls from 1.24 to 0.89, and MIO throttle (the shared-memory instruction queue) becomes an even larger share of each warp's wait.

**Conclusion.** Tile size alone does not break the bottleneck. Smem load wavefronts are identical (3.22 G) because each thread still produces one output, so the shared-memory instruction count per FMA does not change; T=32 only trades less global traffic for less occupancy. T=16 stays the default for 02. The remaining headroom is in raising FMAs per shared-memory load (register tiling, next).

## Roadmap

- [x] 01 naive baseline + benchmark harness
- [x] 02 shared memory tiling + Nsight Compute analysis
- [x] TILE=32 experiment: higher reuse vs lower occupancy (T=16 wins; see above)
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
