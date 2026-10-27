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

| Size | cuBLAS | 01 naive | 02 T=16 | 02 T=32 | 03 1D | 04 2D | 05 vectorized | 06 Bs-conflict-free | 06 vs 05 |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 5.8 TFLOPS | 1088 (18.7%) | 1515 (26.0%) | 1387 (23.8%) | 2497 (42.9%) | 2058 (35.3%) | 2222 (38.1%) | 2300 (39.5%) | 1.04× |
| 1024 | 10.8 TFLOPS | 1210 (11.2%) | 1642 (15.3%) | 1506 (14.0%) | 4191 (38.9%) | 5029 (46.7%) | 6072 (56.4%) | 6423 (59.7%) | 1.06× |
| 2048 | 14.5 TFLOPS | 1432 (9.9%) | 2065 (14.3%) | 1858 (12.8%) | 5825 (40.3%) | 10131 (70.0%) | 11551 (79.9%) | 11844 (81.9%) | 1.03× |
| 4096 | 15.6 TFLOPS | 1326 (8.5%) | 1974 (12.6%) | 1786 (11.4%) | 5358 (34.2%) | 11055 (70.6%) | 12731 (81.4%) | 13068 (83.5%) | 1.03× |

Kernel columns are GFLOPS with % of cuBLAS in brackets. Absolute numbers drift between runs (e.g. 04 at 4096 measured 10.3–11.2 TFLOPS and 05 11.6–12.7 TFLOPS across runs), but 05 / 04 stayed at 1.12–1.14× at 4096. 06 / 05 at 4096 measured 1.03–1.05× in wall-clock runs and 1.06× in Nsight Compute cycles; the gain is small enough that run-to-run noise is of the same order.

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

### 05 — vectorized loads + transposed As (BM = BN = 128, BK = 8, TM = TN = 8)

Same 128×128 tiling as 04, with two changes: global→shared loads use one `float4` (128-bit) per thread, and `As` is stored transposed (`As[BK][BM]`) so a thread's 8 `regM` values are contiguous in shared memory. The writeback to C is also `float4`. The shared arrays are `__align__(16)` and all inner loops carry explicit `#pragma unroll`.

- **Loading**: one `float4` of A per thread (`innerRowA = tid/2`, `innerColA = tid%2`), scattered into `As` as 4 scalar stores (the transpose); one `float4` of B per thread, stored with a single 128-bit store.
- **Requirements**: same size constraints and runtime check as 04; the 16-byte alignment additionally needs K and N multiples of 4 (guaranteed by the 8/128 checks).
- **Resources**: 127 registers/thread, 0 bytes spilled.

### 06 — Bs-conflict-free column split (BM = BN = 128, BK = 8, TM = TN = 8)

An experiment on top of 05 to test where the shared-memory load bank conflicts come from. Instead of owning 8 adjacent columns, each thread owns two groups of 4: columns `threadCol*4 .. +3` and `BN/2 + threadCol*4 .. +3`. Within a warp the 16 distinct `threadCol` values then read contiguous 16-byte chunks (256 contiguous bytes per half) instead of chunks 32 bytes apart. The result mapping is split the same way, and C is written back as two `float4` per row. Everything else (loads, transposed `As`, 127 registers, no spill) is identical to 05.

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

## Profiling: 04 → 05

Nsight Compute, 4096³, one pass each:

| Metric | 04 | 05 |
|---|---|---|
| Elapsed cycles | 19.29 M | 16.98 M (1.14× fewer) |
| Issue slots busy / IPC (active) | 56.5% / 2.31 | 60.0% / 2.44 |
| FMA pipe utilization | 50.0% | 54.8% |
| Registers per thread / spill | 128 / 0 | 127 / 0 |
| Achieved occupancy | 32.6% | 32.6% |
| Global load instructions (executed) | 33.6 M | 8.4 M (4× fewer) |
| Global store instructions (executed) | 0.52 M | 0.13 M |
| Shared store instructions (executed) | 33.6 M | 21.0 M |
| Shared load instructions (executed) | 134 M | 134 M |
| SASS (static): `LDG` / `STG` | 8 `LDG.E` / 64 `STG.E` | 2 `LDG.E.128` / 16 `STG.E.128` |
| SASS (static): `LDS.128` / `STS` | 32 / 8 scalar | 32 / 4 scalar + 1 `STS.128` |
| Smem load wavefronts | 671 M | 671 M |
| **Smem load bank conflicts** | **268,435,456** | **268,435,456** |
| Smem store bank conflicts | 0 | 16.8 M |
| Warp cycles per issued instruction | 6.79 | 6.40 |
| L1/TEX throughput / Mem pipes busy | 72.8% / 36.2% | 83.9% / 33.3% |
| DRAM bytes read | 1.43 GB | 1.28 GB |

Stall breakdown (cycles per issued instruction, share of total in brackets):

| Stall reason | 04 (total 6.79) | 05 (total 6.40) |
|---|---|---|
| MIO throttle | 0.91 (13%) | 0.36 (6%) |
| Long scoreboard | 0.65 (10%) | 1.14 (18%) |
| Barrier | 0.77 (11%) | 0.67 (10%) |
| Short scoreboard | 0.22 (3%) | 1.00 (16%) |
| Not selected | 1.90 (28%) | 1.52 (24%) |
| Dispatch | 0.81 (12%) | 0.49 (8%) |
| Selected (issuing) | 1.00 (15%) | 1.00 (16%) |

### Findings

1. **The gain comes from vectorized global access, not from the transposed `As`.** Executed global-load instructions fell 4× (33.6 M → 8.4 M), global stores 4×, and shared-store instructions 37% (33.6 M → 21.0 M). MIO-throttle and LG-throttle stalls dropped, IPC rose 2.31 → 2.44 and the FMA pipe from 50% to 54.8%, giving 1.14× fewer cycles.
2. **Transposing `As` did not remove the shared-memory load bank conflicts.** 05 has exactly the same 268,435,456 load conflicts, 671 M load wavefronts and 134 M load instructions as 04. In 04 the compiler already vectorized the `As` reads into `LDS.128` (see 03 → 04), so the transposed layout gave no further reduction in load cost. The roadmap item "transposed `As` fixes the 40% conflicts", written after the 04 profile, was wrong.
3. **Since the `As` layout changed and the conflicts did not, the conflicts are probably caused by `Bs` (or by the access pattern shared by both layouts).** Within a warp the 16 distinct `threadCol` values read 16 B chunks at a 32 B stride (`threadCol * TN` floats), which maps several threads onto the same banks. This is a hypothesis consistent with the data; I did not isolate it with per-instruction counters.
4. **The transposed store added a small new cost:** 16.8 M store bank conflicts (04 had none). Each warp writes `As[innerColA*4+k][innerRowA]` with `innerColA` ∈ {0, 1}, two addresses 4 × 128 floats apart that land in the same bank. Short-scoreboard stalls also rose (0.22 → 1.00 cycles); I did not investigate why.
5. **Occupancy and registers are unchanged** (127 vs 128 registers, 33% occupancy, no spill), so this step isolates the effect of the memory-instruction changes.

**Takeaway**: 05 reaches 81% of cuBLAS at 4096 (78.5% at 2048). The shared-memory load conflicts (40% of load wavefronts) remain exactly as before and are the next concrete target; the other candidates are double buffering (long scoreboard + barrier ≈ 28% of stalls) and the register/occupancy trade-off.

## Profiling: 05 → 06

**Hypothesis and test.** The 05 profile showed 268 M load bank conflicts (40% of load wavefronts), unchanged when `As` was transposed (04 → 05). That pointed away from `As` and toward `Bs`, where the 16 distinct `threadCol` values of a warp read 16-byte chunks at a 32-byte stride (`threadCol * 8` floats). 06 changes only that: the `Bs` read pattern (and, as a side effect, the C write pattern). If the hypothesis was right, load conflicts would drop to ~0 while the number of load instructions and the `As` layout stay the same.

Nsight Compute, 4096³, one pass each:

| Metric | 05 | 06 |
|---|---|---|
| **Smem load bank conflicts** | 268,435,456 | **0** |
| **Smem load wavefronts** | 671 M | **402.7 M** (−40%) |
| Smem load instructions | 134 M | 134 M |
| Smem store bank conflicts | 16.8 M | 16.8 M |
| **Elapsed cycles** | 16.97 M | **16.01 M** (1.06× fewer) |
| Issue slots busy / IPC (active) | 60.0% / 2.45 | 63.6% / 2.59 |
| FMA pipe utilization | 54.8% | 58.1% |
| L1/TEX throughput / Mem pipes busy | 84.0% / 33.3% | 59.5% / 35.3% |
| Registers per thread / achieved occupancy | 127 / 32.6% | 127 / 32.6% |
| Warp cycles per issued instruction | 6.40 | 6.03 |

Stall breakdown (cycles per issued instruction, share of total in brackets):

| Stall reason | 05 (total 6.40) | 06 (total 6.03) |
|---|---|---|
| MIO throttle | 0.36 (6%) | 0.18 (3%) |
| Short scoreboard | 1.00 (16%) | 0.61 (10%) |
| Long scoreboard | 1.14 (18%) | 1.09 (18%) |
| Barrier | 0.67 (10%) | 0.67 (11%) |
| Not selected | 1.52 (24%) | 1.71 (28%) |
| Dispatch | 0.49 (8%) | 0.55 (9%) |
| Selected (issuing) | 1.00 (16%) | 1.00 (17%) |

### Findings

1. **The hypothesis is confirmed: the load conflicts came from the `Bs` access pattern.** Load conflicts went from 268 M to exactly 0 with load instruction count and `As` layout unchanged, so the `Bs` read pattern is the only thing that can explain it. The remaining 402.7 M load wavefronts equal 134.2 M instructions × 3, i.e. 3 wavefronts per `LDS.128` and no conflict replays (the "×3" breakdown is my arithmetic, not checked per instruction).
2. **The speedup is small (1.06× in cycles, 1.03–1.05× in wall-clock) because shared memory was not the bottleneck.** Removing 40% of the load wavefronts took L1/TEX throughput from 84% to 59%, but the memory pipes were only ~33–35% busy before and after. The saving shows up as fewer MIO-throttle and short-scoreboard stalls (0.36 → 0.18 and 1.00 → 0.61 cycles), a modest IPC gain (2.45 → 2.59) and FMA pipe 54.8% → 58.1%. Long-scoreboard and barrier stalls, which dominate what is left, did not change.
3. **Confounder: the C write pattern changed too.** Splitting the columns also changed how results are written back (two `float4` per row, 64 columns apart). The shared-memory metrics are unaffected by this, so finding 1 stands, but the cycle improvement is the combined effect of the `Bs` read fix and the new write pattern. I did not run a variant that changes only the write-back to separate them; its share of total time is small.
4. **Unchanged by design:** registers (127), occupancy (32.6%), no spill, and the 16.8 M store conflicts from the transposed `As` writes.

## Remaining bottlenecks / future work

Based on the 06 profile (issue slots 63.6%, FMA pipe 58.1%, 83.5% of cuBLAS at 4096). None of the items below has been implemented or measured yet:

- **Long scoreboard (18% of stall cycles) and barrier (11%)**: global-load latency is not overlapped with compute, and every K-step waits on two `__syncthreads()`. Double buffering (load the next tile into a second shared-memory buffer while computing on the current one) should hide both and halve the barrier count per tile.
- **16.8 M `As` store bank conflicts**: the transposed scalar stores of `As` collide in pairs. Padding the `As` rows (for example `BM + 4` floats, which keeps 16-byte alignment for the `LDS.128` reads) is the usual fix. Whether it pays off is uncertain since stores are a small share of shared-memory traffic.
- **Warp tiling**: add an explicit warp-level tile between the block tile and the thread tile so each warp's shared-memory reads and register footprint are organized deliberately, which also gives more freedom in choosing the thread tile.
- **Parameter auto-tuning**: BM/BN/BK/TM/TN are fixed at 128/128/8/8/8. The register count (127) caps occupancy at 33%; a sweep over tile shapes, `__launch_bounds__` and register limits, possibly per problem size (512 under-fills the GPU with 16 blocks), would show whether another point on the occupancy / ILP trade-off is faster.

## Roadmap

- [x] 01 naive baseline + benchmark harness
- [x] 02 shared memory tiling + Nsight Compute analysis
- [x] TILE=32 experiment: higher reuse vs lower occupancy (T=16 wins; see above)
- [x] 03 1D register blocktiling (BM=BN=64, BK=8, TM=8) + Nsight Compute comparison
- [x] 04 2D register blocktiling (BM=BN=128, BK=8, TM=TN=8) + Nsight Compute comparison (69% of cuBLAS at 4096)
- [x] 05 `float4` global loads + transposed `As` (1.13× over 04; did *not* reduce the smem load conflicts)
- [x] 06 split TN into two `float4` halves: confirmed the smem load conflicts came from `Bs`, removed them (1.03–1.06× over 05)
- [ ] 07 double buffering (long scoreboard + barrier stalls)
- [ ] 08 pad `As` (store conflicts), warp tiling, parameter auto-tuning; revisit register count / occupancy trade-off
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
