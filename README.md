# cuda-kernels

CUDA performance-optimization practice (target: RTX 3080 Ti Laptop, sm_86).

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
