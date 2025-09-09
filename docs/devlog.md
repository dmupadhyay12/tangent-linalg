### 2025-08-10 — Naive Matmul Implemented

- Implemented operator\* for Matrix<T, R, C> with static dimension check.
- Verified correctness for small test cases (2×2, 3×3).
- Uses triple loop (i-j-k), no blocking/tiling, no SIMD.
- **Status:** Correct but unoptimized. Benchmarking TBD.
- **Next:** Add support for mixed types, and perform benchmarking with Eigen

### First Implementation with results:

template <typename T_rhs, size_t rows_rhs, size_t cols_rhs>
auto operator*(const Matrix<T_rhs, rows_rhs, cols_rhs> &rhs) const {
static_assert(
columns == rows_rhs,
"Matrix multiplication cannot be done due to incompatible dimensions");
Matrix<T, rows, cols_rhs> output;
for (size_t col = 0; col < cols_rhs; col++) {
for (size_t row = 0; row < rows; row++) {
T value = 0;
for (size_t col_lhs = 0; col_lhs < columns; col_lhs++) {
value += (*this)(row, col_lhs) \* rhs(col_lhs, col);
}
output(row, col) = value;
}
}
return output;
}

BM_Tangent<3, 3, 3> 35.3 ns 35.3 ns 17713590 GFLOP/s=1.52776G/s
BM_Eigen<3, 3, 3> 37.6 ns 37.6 ns 16512949 GFLOP/s=1.43671G/s
BM_Tangent<64, 64, 64> 375467 ns 375422 ns 1769 GFLOP/s=1.39653G/s
BM_Eigen<64, 64, 64> 74020 ns 74016 ns 8440 GFLOP/s=7.08343G/s
BM_Tangent<128, 128, 128> 3832954 ns 3832667 ns 182 GFLOP/s=1094.36M/s
BM_Eigen<128, 128, 128> 541433 ns 541408 ns 1103 GFLOP/s=7.74703G/s

### Second implementation

static*assert(
columns == rows_rhs,
"Matrix multiplication cannot be done due to incompatible dimensions");
Matrix<T, rows, cols_rhs> output;
for (size_t row = 0; row < rows; row++) {
for (size_t col = 0; col < cols_rhs; col++) {
T value = 0;
for (size_t col_lhs = 0; col_lhs < columns; col_lhs++) {
value += (\_this)(row, col_lhs) * rhs(col_lhs, col);
}
output(row, col) = value;
}
}

BM_Tangent<3, 3, 3> 29.6 ns 29.6 ns 23098908 GFLOP/s=1.82561G/s
BM_Eigen<3, 3, 3> 41.7 ns 41.7 ns 16377185 GFLOP/s=1.29386G/s
BM_Tangent<64, 64, 64> 422827 ns 422812 ns 1561 GFLOP/s=1.24G/s
BM_Eigen<64, 64, 64> 90694 ns 90686 ns 6718 GFLOP/s=5.78138G/s
BM_Tangent<128, 128, 128> 4509896 ns 4509897 ns 136 GFLOP/s=930.022M/s
BM_Eigen<128, 128, 128> 705519 ns 705460 ns 861 GFLOP/s=5.94549G/s
