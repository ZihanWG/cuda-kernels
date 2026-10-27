cmake_minimum_required(VERSION 3.24)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)   # RTX 3080 Ti Laptop (Ampere)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# -lineinfo lets Nsight Compute map SASS back to source lines
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_library(gemm_kernels STATIC
  src/gemm/cublas_ref.cu
  src/gemm/gemm_01_naive.cu
  src/gemm/gemm_02_smem_tiling.cu
)
target_include_directories(gemm_kernels PUBLIC src)
target_link_libraries(gemm_kernels PUBLIC CUDA::cudart CUDA::cublas)

add_executable(bench_gemm bench/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE gemm_kernels)
