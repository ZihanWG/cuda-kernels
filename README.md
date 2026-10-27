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

| Size | cuBLAS | 01 naive | 02 T=16 | 02 T=32 | 03 1D | 04 2D | 04 vs 03 |
|---|---|---|---|---|---|---|---|
| 512 | 5.7 TFLOPS | 1088 (19.1%) | 1520 (26.6%) | 1394 (24.4%) | 2521 (44.2%) | 2065 (36.2%) | 0.82× |
| 1024 | 9.3 TFLOPS | 1206 (13.0%) | 1636 (17.6%) | 1502 (16.1%) | 4186 (45.0%) | 5001 (53.7%) | 1.19× |
| 2048 | 11.6 TFLOPS | 1398 (12.0%) | 2013 (17.3%) | 1869 (16.0%) | 5844 (50.2%) | 9659 (82.9%) | 1.65× |
| 4096 | 15.7 TFLOPS | 1333 (8.5%) | 1963 (12.5%) | 1786 (11.4%) | 5301 (33.8%) | 10807 (68.9%) | 2.04× |

Kernel columns are GFLOPS with % of cuBLAS in brackets. In this run the cuBLAS baseline at 1024 and 2048 was lower than in neighbouring runs (e.g. 9.3 vs 10.5–10.8 TFLOPS at 1024), which inflates those two percentages; the 4096 baseline is normal.

All numbers are from one interleaved run. The laptop GPU throttles over consecutive runs: in repeated 4096 runs taken while throttled, the T=32 / T=16 time ratio ranged from 0.96× to 1.37×, so close wall-clock comparisons at 4096 (02 T=16 vs T=32) are not reliable; the cycle counts from Nsight Compute are the evidence used for conclusions.

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

### 03 — 1D register blocktiling (BM = BN = 64, BK = 8, TM = 8)

Each block of 512 threads computes a 64×64 tile of C; each thread computes a column of 8 outputs (TM = 8) and keeps the 8 accumulators in registers. Per K-step a thread reads one `Bs` value into a register and multiplies it with 8 `As` values, so every shared-memory load of `Bs` feeds 8 FMAs (≈1.1 smem loads per FMA instead of ≈2 in 02).

- **Loading**: `As` (64×8) and `Bs` (8×64) each hold 512 elements, exactly one per thread, so the cooperative load needs no loop.
- **Bank conflicts**: `Bs[dot][threadCol]` is 32 consecutive floats per warp; `As[threadRow*TM+i][dot]` is a broadcast (all 32 lanes of a warp share `threadRow`); `As` stores map address = `tid`. All conflict-free for loads.
- **Size requirement**: M, N must be multiples of 64 and K of 8. This is checked at runtime (not `assert`, which Release builds compile out); on violation the wrapper prints an error to stderr and skips the launch, so the benchmark's NaN pre-fill makes the check FAIL instead of silently computing garbage.

### 04 — 2D register blocktiling (BM = BN = 128, BK = 8, TM = TN = 8)

Each block of 256 threads computes a 128×128 tile of C; each thread owns an 8×8 sub-block (64 accumulators in registers). Per K-step (`dot`) a thread loads 8 values of `As` into `regM` and 8 values of `Bs` into `regN`, then does the 8×8 outer product: 16 shared-memory reads feed 64 FMAs (0.25 smem reads per FMA, vs ≈1.1 in 03 and 2 in 02).

- **Loading**: 256 threads cover the 128×8 `As` tile in 4 strided passes (`strideA = 256/BK = 32`) and the 8×128 `Bs` tile in 4 passes (`strideB = 256/BN = 2`).
- **Size requirement**: M, N multiples of 128 and K a multiple of 8, checked at runtime (prints to stderr and skips the launch, so the benchmark's NaN pre-fill makes it FAIL rather than silently wrong).
- **Resources**: 128 registers/thread, 0 bytes spilled, 8 KB static shared memory per block.

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

**Prediction.** My prediction: slower — occupancy drops from ~100% to 66.7%, fewer warps to hide latency.

Alternative hypothesis: barrier stalls with only one block per SM — refuted by WarpStateStats (barrier share dropped from 15% to 7%).

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

**The barrier hypothesis was refuted.** Barrier stalls are *smaller* with T=32 (7% vs 15% of stall cycles); a larger block spends fewer barrier rounds per useful FMA. The loss comes from latency hiding, as predicted: with 8 instead of 12 warps per scheduler there are fewer eligible warps (1.06 vs 1.32), IPC falls from 1.24 to 0.89, and MIO throttle (the shared-memory instruction queue) becomes an even larger share of each warp's wait.

**Conclusion.** Tile size alone does not break the bottleneck. Smem load wavefronts are identical (3.22 G) because each thread still produces one output, so the shared-memory instruction count per FMA does not change; T=32 only trades less global traffic for less occupancy. T=16 stays the default for 02. The remaining headroom is in raising FMAs per shared-memory load (register tiling, next).

## Profiling: 02 → 03

Nsight Compute, 4096³, one pass each (02 = T=16):

| Metric | 02 T=16 | 03 1D blocktile |
|---|---|---|
| Elapsed cycles | 114.5 M | 39.3 M (2.91× fewer) |
| L1/TEX (LSU) throughput | 97.7% | 83.1% |
| DRAM bytes read | 17.54 GB | 4.41 GB |
| DRAM throughput | 42.0% | 31.0% |
| L2 throughput | 31.2% | 22.9% |
| Issue slots busy | 29.7% | 37.6% |
| IPC (active) | 1.24 | 1.51 |
| Achieved occupancy | 99.8% | 66.2% |
| Registers per thread | 38 | 47 |
| Smem load wavefronts | 3.222 G | 1.342 G |
| Smem load / store bank conflicts | 252 K / 12.9 M | 74 K / 6.2 M |
| Active / eligible warps per scheduler | 11.97 / 1.32 | 7.94 / 1.21 |
| Warp cycles per issued instruction | 38.5 | 21.0 |

Stall breakdown (cycles per issued instruction, share of total in brackets):

| Stall reason | 02 T=16 | 03 1D blocktile |
|---|---|---|
| MIO throttle (smem queue) | 18.6 (48%) | 6.8 (32%) |
| Long scoreboard (global load latency) | 7.1 (18%) | 5.2 (25%) |
| Barrier | 5.6 (15%) | 2.1 (10%) |
| Short scoreboard (smem latency) | 0.5 (1%) | 2.0 (10%) |
| Other (not selected, wait, dispatch, …) | ~6.7 | ~5.0 |

### Findings

1. **The speedup comes from fewer shared-memory instructions per FMA.** Smem load wavefronts drop 2.4× for the same FLOPs (the register-reuse effect, plus whatever vectorization the compiler applied), and MIO-throttle stalls fall from 18.6 to 6.8 cycles per instruction. LSU throughput leaves saturation (98% → 83%) and warps spend 21 instead of 38.5 cycles between issues; this translates into the 2.9× cycle reduction.
2. **Global traffic also fell 4×**, as expected from the 64×64 block (reuse 64 vs 16): DRAM reads 17.5 → 4.4 GB, L2 throughput 31% → 23%.
3. **Occupancy went down on purpose and it was a fair trade.** 47 registers × 512 threads fit 2 blocks per SM (66.7%); ncu reports registers as the limiter. A 03 variant below ~40 registers would fit 3 blocks (by my arithmetic, not measured).
4. **The bottleneck is spread out now.** LSU is still the busiest unit (82.6% vs FMA 24.3%), but no single stall dominates: MIO 32%, global-load latency 25%, barrier 10%, smem latency 10%. Issue slots are only 37.6% busy, so the FMA pipes are still mostly idle.

**Takeaway**: 1D blocktiling reached 34–43% of cuBLAS. The remaining levers are more FMAs per smem load (2D register tiling, each thread a TM×TN block, loading `As` fragments into registers too), hiding global-load latency (double buffering), and wider/better-coalesced loads (`float4`; the 8-float-wide `As` row read is only 32 B per row).

## Profiling: 03 → 04

Nsight Compute, 4096³, one pass each:

| Metric | 03 1D | 04 2D |
|---|---|---|
| Elapsed cycles | 39.7 M | 19.3 M (2.06× fewer) |
| Issue slots busy / IPC (active) | 37.2% / 1.51 | 55.3% / 2.31 |
| FMA pipe utilization | 24.1% | 49.0% |
| Achieved / theoretical occupancy | 66.2% / 66.7% | 32.6% / 33.3% |
| Registers per thread | 47 | 128 |
| Active / eligible warps per scheduler | 7.94 / 1.21 | 3.91 / 1.67 |
| Warp cycles per issued instruction | 21.0 | 6.8 |
| Local memory load/store bytes | 0 / 0 | 0 / 0 |
| Smem load instructions | 805 M | 134 M |
| Smem load wavefronts | 1342 M | 671 M |
| Smem load bank conflicts | 79 K | 268 M (40% of wavefronts) |
| Smem store bank conflicts | 6.0 M | 0 |
| L1/TEX throughput / Mem pipes busy | 83.0% / 81.7% | 72.8% / 35.4% |
| DRAM bytes read / DRAM throughput | 4.46 GB / 31.5% | 1.45 GB / 21.4% |
| L2 throughput | 23.4% | 26.4% |

Stall breakdown (cycles per issued instruction, share of total in brackets):

| Stall reason | 03 (total 21.0) | 04 (total 6.8) |
|---|---|---|
| MIO throttle | 6.8 (33%) | 0.9 (13%) |
| Long scoreboard | 5.2 (25%) | 0.7 (10%) |
| Barrier | 2.0 (10%) | 0.8 (11%) |
| Short scoreboard | 2.0 (10%) | 0.2 (3%) |
| Not selected | 2.2 (10%) | 1.9 (28%) |
| Dispatch | 0.6 (3%) | 0.8 (12%) |
| Selected (issuing) | 1.0 (5%) | 1.0 (15%) |

### Findings

1. **Occupancy halved (66% → 33%) yet IPC rose 1.5× and cycles halved.** Occupancy is a means, not the goal. With 128 registers only 2 blocks (16 warps) fit per SM, but each warp now has 64 independent FMAs per 16 shared-memory reads, so a warp rarely waits: stall time per issued instruction fell from 21 to 6.8 cycles and 1.67 of ~3.9 warps per scheduler are eligible on average. The extra thread-level parallelism that 03 needed to hide latency is replaced by instruction-level parallelism inside each thread. (This reading fits the stall data; I did not run an experiment that isolates it, e.g. forcing lower register use.)
2. **No register spilling despite 128 registers.** `ptxas -v` reports 0 bytes of stack frame and spill; ncu shows 0 local-memory load/store bytes and 0 spilling requests. The 64 accumulators + 16 operand registers stay in registers because all loops have compile-time bounds and unroll fully. Note the outer-product loops carry no explicit `#pragma unroll`; they rely on the compiler, which is worth re-checking if TM/TN change.
3. **The compiler generated `LDS.128` on its own.** The SASS has 32 `LDS.128` and no scalar `LDS`: with the `dot` loop unrolled, each thread's 8 values of a row of `As` and of `Bs` are contiguous, so they merge into 128-bit loads. Shared-memory load instructions fell 6× (805 M → 134 M) and wavefronts 2×, which is why MIO throttle stalls dropped from 33% to 13% of the total.
4. **Bank conflicts appeared and are now the main waste on the shared-memory side.** 268 M of 671 M load wavefronts (40%) are conflict replays, while stores are conflict-free. My pre-run estimate of 2-way (`As`) and 4-way (`Bs`) conflicts assumed scalar loads; with `LDS.128` the conflict rules differ and this data cannot split the conflicts between `As` and `Bs`. The shared-memory pipes are only 35% busy, so conflicts are not the bottleneck at the moment.
5. **At 512 the grid is too small for 04.** 128×128 blocks give only 16 blocks for 58 SMs (03's 64×64 blocks give 64), so most SMs idle and 04 is 0.82× of 03 there. 04 pays off from 1024 up and reaches 1.65–2.04× at 2048–4096.

**Takeaway**: 04 reaches 69% of cuBLAS at 4096 (83% at 2048). About 85% of issued instructions are FFMA (estimated from IPC × cycles), the FMA pipe is at 49%, and issue slots at 55%. Remaining candidates, none yet measured: removing the 40% bank-conflict overhead (e.g. transposed `As`), hiding global-load latency with double buffering (long scoreboard + barrier still ≈ 21% of stalls), `float4` global loads, and tuning register use to raise occupancy.

## Roadmap

- [x] 01 naive baseline + benchmark harness
- [x] 02 shared memory tiling + Nsight Compute analysis
- [x] TILE=32 experiment: higher reuse vs lower occupancy (T=16 wins; see above)
- [x] 03 1D register blocktiling (BM=BN=64, BK=8, TM=8) + Nsight Compute comparison
- [x] 04 2D register blocktiling (BM=BN=128, BK=8, TM=TN=8) + Nsight Compute comparison (69% of cuBLAS at 4096)
- [ ] 05 fix the `As` bank conflicts (transposed `As`) and use `float4` global loads
- [ ] 06 double buffering; revisit register count / occupancy trade-off
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
