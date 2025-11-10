// tests/matrix_tests.cpp
#define CATCH_CONFIG_MAIN
#include "tangent/matrix.hpp"
#include "tangent/vector.hpp"
#include <catch2/catch_all.hpp>
#include <limits>
#include <random>

using Catch::Approx;
using tangent::Matrix;

// Helper functions
template <typename T, size_t R, size_t C>
static Matrix<T, R, C> from_init(std::initializer_list<T> vals) {
  REQUIRE(vals.size() == R * C);
  std::array<T, R * C> a{};
  size_t idx = 0;
  for (auto v : vals)
    a[idx++] = v;
  return Matrix<T, R, C>(a);
}

// simple reference multiply (truth)
template <typename L, size_t R, size_t K, typename RhsT, size_t K2, size_t C>
auto reference_mul(const Matrix<L, R, K> &A, const Matrix<RhsT, K2, C> &B) {
  static_assert(K == K2, "dims");
  using OutT = decltype(std::declval<L>() * std::declval<RhsT>());
  Matrix<OutT, R, C> out;
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j) {
      OutT s{};
      for (size_t k = 0; k < K; ++k)
        s += A(i, k) * B(k, j);
      out(i, j) = s;
    }
  return out;
}

template <typename T, size_t R, size_t C>
static Matrix<T, R, C> from_list(std::initializer_list<T> vals) {
  REQUIRE(vals.size() == R * C);
  std::array<T, R * C> a{};
  size_t i = 0;
  for (auto v : vals)
    a[i++] = v;
  return Matrix<T, R, C>(a);
}

TEST_CASE("Add: small fixed cases") {
  auto A = from_init<int, 2, 2>({1, 2, 3, 4});
  auto B = from_init<int, 2, 2>({5, 6, 7, 8});
  auto C = A + B;
  REQUIRE(C(0, 0) == 6);
  REQUIRE(C(0, 1) == 8);
  REQUIRE(C(1, 0) == 10);
  REQUIRE(C(1, 1) == 12);
}

TEST_CASE("Mul: 2x2 hand check") {
  auto A = from_init<int, 2, 2>({1, 2, 3, 4});
  auto B = from_init<int, 2, 2>({5, 6, 7, 8});
  auto C = A * B;         // returns Matrix<T=left,2,2> = int
  REQUIRE(C(0, 0) == 19); // 1*5 + 2*7
  REQUIRE(C(0, 1) == 22); // 1*6 + 2*8
  REQUIRE(C(1, 0) == 43); // 3*5 + 4*7
  REQUIRE(C(1, 1) == 50); // 3*6 + 4*8
}

TEST_CASE("1xN * Nx1 (dot product)") {
  auto A = from_list<int, 1, 4>({1, 2, 3, 4});
  auto B = from_list<int, 4, 1>({5, 6, 7, 8});
  auto got = A * B; // 1x1
  auto ref = reference_mul(A, B);
  REQUIRE(got(0, 0) == ref(0, 0)); // 1*5 + 2*6 + 3*7 + 4*8 = 70
  REQUIRE(got(0, 0) == 70);
}

TEST_CASE("3x1 * 1x4 (row/col expansion)") {
  auto A = from_list<float, 3, 1>({2.0f, -1.0f, 0.5f});
  auto B = from_list<float, 1, 4>({10.0f, 0.0f, -3.0f, 4.0f});
  auto got = A * B; // 3x4
  auto ref = reference_mul(A, B);
  for (size_t r = 0; r < 3; ++r)
    for (size_t c = 0; c < 4; ++c)
      REQUIRE(got(r, c) == Approx(ref(r, c)).epsilon(1e-6f));
}

TEST_CASE("Random 3x4 * 4x5 (float)") {
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

  Matrix<float, 3, 4> A;
  Matrix<float, 4, 5> B;
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 4; ++j)
      A(i, j) = dist(rng);
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 5; ++j)
      B(i, j) = dist(rng);

  auto got = A * B; // 3x5
  auto ref = reference_mul(A, B);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 5; ++j)
      REQUIRE(got(i, j) == Approx(ref(i, j)).epsilon(1e-5f).margin(1e-6f));
}

TEST_CASE("Identity and zero (mixed sizes)") {
  Matrix<double, 4, 4> I;
  for (size_t i = 0; i < 4; ++i)
    I(i, i) = 1.0;

  Matrix<double, 4, 3> A;
  size_t v = 1;
  for (size_t r = 0; r < 4; ++r)
    for (size_t c = 0; c < 3; ++c)
      A(r, c) = double(v++);

  Matrix<double, 4, 4> Z; // zeros

  auto AI = I * A; // 4x3
  auto ZA = Z * A; // 4x3 zeros

  for (size_t r = 0; r < 4; ++r)
    for (size_t c = 0; c < 3; ++c) {
      REQUIRE(AI(r, c) == Approx(A(r, c)));
      REQUIRE(ZA(r, c) == Approx(0.0));
    }
}

TEST_CASE("Mixed types promote correctly (double * float)") {
  auto A = from_list<double, 2, 3>({1.5, -2.0, 0.25, 4.0, 5.0, -3.0});
  auto B = from_list<float, 3, 2>({0.5f, 1.0f, -2.0f, 0.25f, 1.25f, -1.0f});

  // NOTE: your current operator* returns Matrix<T_left,...>.
  // If left T is double, result is double → safe here.
  auto got = A * B;
  auto ref = reference_mul(A, B);
  for (size_t r = 0; r < 2; ++r)
    for (size_t c = 0; c < 2; ++c)
      REQUIRE(got(r, c) == Approx(ref(r, c)).epsilon(1e-12).margin(1e-12));
}

TEST_CASE("Scalar multiplication of a matrix by a scalar, scalar second") {
  auto A = from_list<double, 3, 3>({2, 4, 6, 8, 10, 12, 14, 16, 18});

  auto res = A * 0.5;

  auto ref = from_list<double, 3, 3>({1, 2, 3, 4, 5, 6, 7, 8, 9});

  for (size_t r = 0; r < 2; r++) {
    for (size_t c = 0; c < 2; c++) {
      REQUIRE(res(r, c) == Approx(ref(r, c)).epsilon(1e-12).margin(1e-12));
    }
  }
}

TEST_CASE("Scalar multiplication of a matrix by a scalar, scalar first") {
  auto A = from_list<double, 3, 3>({2, 4, 6, 8, 10, 12, 14, 16, 18});

  auto res = 0.5 * A;

  auto ref = from_list<double, 3, 3>({1, 2, 3, 4, 5, 6, 7, 8, 9});

  for (size_t r = 0; r < 2; r++) {
    for (size_t c = 0; c < 2; c++) {
      REQUIRE(res(r, c) == Approx(ref(r, c)).epsilon(1e-12).margin(1e-12));
    }
  }
}

TEST_CASE(
    "Scalar multiplication of a matrix of floats by a scalar, scalar first") {
  auto A =
      from_list<double, 3, 3>({0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8});

  auto res = 0.5 * A;

  auto ref =
      from_list<double, 3, 3>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9});

  for (size_t r = 0; r < 2; r++) {
    for (size_t c = 0; c < 2; c++) {
      REQUIRE(res(r, c) == Approx(ref(r, c)).epsilon(1e-12).margin(1e-12));
    }
  }
}

TEST_CASE("Validate equality operator for matrices") {
  auto A = from_list<int, 1, 4>({1, 2, 3, 4});

  auto ref = from_list<int, 1, 4>({1, 2, 3, 4});

  REQUIRE(A == ref);
}

TEST_CASE("Validate column vector generation") {
  auto vector_form = tangent::Vec<int, 4>({1, 2, 3, 4});
  auto matrix_form = from_list<int, 1, 4>({1, 2, 3, 4});

  REQUIRE(vector_form == matrix_form);
}

TEST_CASE("Validate row vector generation") {
  auto vector_form = tangent::RowVec<int, 4>({1, 2, 3, 4});
  auto matrix_form = from_list<int, 4, 1>({1, 2, 3, 4});

  REQUIRE(vector_form == matrix_form);
}

TEST_CASE("Validate dot product standard") {
  auto vector_1 = tangent::Vec<int, 4>({1, 2, 3, 4});
  auto vector_2 = tangent::Vec<int, 4>({2, 3, 4, 5});

  auto dot_product = tangent::dot(vector_1, vector_2);

  // The expected dot product is 2 + 6 + 12 + 20 = 40
  REQUIRE(dot_product == 40);
}

TEST_CASE("Validate dot product different_types") {
  auto vector_1 = tangent::Vec<int, 4>({1, 2, 3, 4});
  auto vector_2 = tangent::Vec<float, 4>({2.5, 3.5, 4.5, 5.5});

  auto dot_product = tangent::dot(vector_1, vector_2);

  // The expected dot product is 2.5 + 7 + 10.5 + 22 = 45.0
  REQUIRE(dot_product == 45.0f);
}

TEST_CASE("Validate norm of a vector") {
  auto vector_1 = tangent::Vec<int, 4>({1, 2, 3, 4});

  auto vec_norm = tangent::norm<int, 4>(vector_1);

  REQUIRE(vec_norm == Approx(5.477225575f).epsilon(1e-6));
}

TEST_CASE("Validate norm of a vector of floats") {
  auto vector_1 = tangent::Vec<float, 4>({1.5, 2.5, 3.5, 4.5});

  auto vec_norm = tangent::norm<float, 4>(vector_1);

  REQUIRE(vec_norm == Approx(6.4031242374).epsilon(1e-6));
}

TEST_CASE("Validate that transpose of a matrix works") {
  auto matrix_1 =
      from_list<float, 3, 2>({0.5f, 1.0f, -2.0f, 0.25f, 1.25f, -1.0f});

  auto matrix_2 = matrix_1.transpose();

  auto matrix_ref = from_list<float, 2, 3>({0.5, -2.0, 1.25, 1.0, 0.25, -1.0});

  for (size_t r = 0; r < 2; ++r)
    for (size_t c = 0; c < 3; ++c)
      REQUIRE(matrix_2(r, c) ==
              Approx(matrix_ref(r, c)).epsilon(1e-12).margin(1e-12));
}