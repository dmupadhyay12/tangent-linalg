#include "include/matrix.hpp"
#include <Eigen/Dense>
#include <benchmark/benchmark.h>

template <int M, int K, int N> static void BM_Tangent(benchmark::State &s) {
  tangent::Matrix<double, M, K> A;
  tangent::Matrix<double, K, N> B;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < K; j++)
      A(i, j) = (i + j) % 7 + 0.1;
  for (int i = 0; i < K; i++)
    for (int j = 0; j < N; j++)
      B(i, j) = (i * 2 - j) % 5 + 0.2;

  const double flops = 2.0 * M * K * N;
  for (auto _ : s) {
    benchmark::DoNotOptimize(A);
    benchmark::DoNotOptimize(B);
    auto C = A * B;
    benchmark::DoNotOptimize(C);
    benchmark::ClobberMemory();
  }
  s.counters["GFLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

template <int M, int K, int N> static void BM_Eigen(benchmark::State &s) {
  Eigen::setNbThreads(1);
  Eigen::Matrix<double, M, K> A;
  Eigen::Matrix<double, K, N> B;
  Eigen::Matrix<double, M, N> C;
  A.setRandom();
  B.setRandom();
  const double flops = 2.0 * M * K * N;
  for (auto _ : s) {
    benchmark::DoNotOptimize(A);
    benchmark::DoNotOptimize(B);
    C.noalias() = A * B;
    benchmark::DoNotOptimize(C);
    benchmark::ClobberMemory();
  }
  s.counters["GFLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

BENCHMARK(BM_Tangent<3, 3, 3>);
BENCHMARK(BM_Eigen<3, 3, 3>);
BENCHMARK(BM_Tangent<64, 64, 64>);
BENCHMARK(BM_Eigen<64, 64, 64>);
BENCHMARK(BM_Tangent<128, 128, 128>);
BENCHMARK(BM_Eigen<128, 128, 128>);
BENCHMARK_MAIN();
