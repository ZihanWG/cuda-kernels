// GEMM benchmark: correctness vs cuBLAS, GFLOPS, and % of cuBLAS.
//
// Laptop GPUs throttle, so absolute GFLOPS drift. Each round runs cuBLAS and
// every kernel back-to-back (same thermal state), and we report the median
// over rounds. "% cuBLAS" = median(cublas time) / median(kernel time).
//
// Usage: bench_gemm [--sizes 512,1024,2048,4096] [--kernels naive,...]
//                   [--warmup 5] [--iters 20]
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "common/cuda_utils.cuh"
#include "gemm/gemm.h"

struct Kernel { const char* name; GemmFn fn; };
static const Kernel kKernels[] = {
    {"cublas", gemm_cublas},
    {"01_naive", gemm_01_naive},
    {"02_smem_tiling", gemm_02_smem_tiling},
};

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out; std::stringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(t);
  return out;
}

static float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static void fill_random(std::vector<float>& v, unsigned seed) {
  unsigned s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = ((s >> 8) & 0xFFFF) / 32768.f - 1.f;  // [-1, 1)
  }
}

int main(int argc, char** argv) {
  std::vector<int> sizes = {512, 1024, 2048, 4096};
  std::vector<std::string> filter;
  int warmup = 5, iters = 20;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--sizes") && i + 1 < argc) {
      sizes.clear();
      for (auto& t : split(argv[++i])) sizes.push_back(std::stoi(t));
    } else if (!strcmp(argv[i], "--kernels") && i + 1 < argc) {
      filter = split(argv[++i]);
    } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1;
    }
  }

  std::vector<Kernel> ks = {kKernels[0]};  // cuBLAS is always the baseline
  for (size_t i = 1; i < sizeof(kKernels) / sizeof(kKernels[0]); ++i) {
    bool take = filter.empty();
    for (auto& f : filter) if (strstr(kKernels[i].name, f.c_str())) take = true;
    if (take) ks.push_back(kKernels[i]);
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("warmup=%d iters=%d (median)\n\n", warmup, iters);
  printf("%6s  %-14s %10s %10s %9s  %s\n", "size", "kernel", "ms", "GFLOPS",
         "%cuBLAS", "check");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  for (int n : sizes) {
    const int M = n, N = n, K = n;
    size_t bytesA = (size_t)M * K * 4, bytesB = (size_t)K * N * 4,
           bytesC = (size_t)M * N * 4;
    std::vector<float> hA(M * (size_t)K), hB(K * (size_t)N);
    fill_random(hA, 1); fill_random(hB, 2);
    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, bytesA)); CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC)); CUDA_CHECK(cudaMalloc(&dRef, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    // Reference + correctness (err normalised by max |ref|; FP32 summation
    // order differs between implementations so exact equality is not expected).
    gemm_cublas(dA, dB, dRef, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> ref(M * (size_t)N), out(M * (size_t)N);
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, bytesC, cudaMemcpyDeviceToHost));
    float refMax = 0.f;
    for (float x : ref) refMax = std::max(refMax, std::fabs(x));

    std::vector<bool> ok(ks.size(), true);
    std::vector<float> err(ks.size(), 0.f);
    for (size_t k = 1; k < ks.size(); ++k) {
      CUDA_CHECK(cudaMemset(dC, 0xFF, bytesC));  // NaN fill catches unwritten outputs
      ks[k].fn(dA, dB, dC, M, N, K, stream);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy(out.data(), dC, bytesC, cudaMemcpyDeviceToHost));
      float maxErr = 0.f;
      for (size_t i = 0; i < out.size(); ++i) {
        float d = std::fabs(out[i] - ref[i]);
        if (!(d <= maxErr)) maxErr = std::isnan(d) ? INFINITY : d;  // NaN-safe
      }
      err[k] = maxErr / refMax;
      ok[k] = err[k] < 1e-3f;
    }

    // Warm-up (also lets clocks ramp up), then interleaved timed rounds.
    for (int w = 0; w < warmup; ++w)
      for (auto& k : ks) k.fn(dA, dB, dC, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::vector<float>> times(ks.size());
    for (int it = 0; it < iters; ++it) {
      for (size_t k = 0; k < ks.size(); ++k) {
        CUDA_CHECK(cudaEventRecord(e0, stream));
        ks[k].fn(dA, dB, dC, M, N, K, stream);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        times[k].push_back(ms);
      }
    }

    double flops = 2.0 * M * N * K;
    float base = median(times[0]);
    for (size_t k = 0; k < ks.size(); ++k) {
      float ms = median(times[k]);
      char chk[64];
      if (k == 0) snprintf(chk, sizeof chk, "baseline");
      else snprintf(chk, sizeof chk, "%s (err %.1e)", ok[k] ? "PASS" : "FAIL", err[k]);
      printf("%6d  %-14s %10.3f %10.1f %8.1f%%  %s\n", n, ks[k].name, ms,
             flops / (ms * 1e-3) / 1e9, 100.0 * base / ms, chk);
    }
    printf("\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
  }
  return 0;
}
