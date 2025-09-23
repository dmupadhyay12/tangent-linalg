#include "matrix.hpp"
#include <array>
#include <iostream>
#include <math.h>

#pragma once

namespace tangent {
// Alias for vectors
template <typename T, size_t N> using Vector = Matrix<T, 1, N>;

template <typename T, size_t N> using RowVector = Matrix<T, N, 1>;

/**
 * @brief Convenience function for constructing column vectors
 *
 * @tparam T
 * @tparam N
 * @param elements
 * @return Vector<T, N>
 */
template <typename T, size_t N>
Vector<T, N> Vec(std::initializer_list<T> elements) {
  std::array<T, N> temp_arr;
  size_t i = 0;
  for (T element : elements) {
    temp_arr[i++] = element;
  }

  return Vector<T, N>(temp_arr);
}

/**
 * @brief Convenience function for constructing row vectors
 *
 * @tparam T
 * @tparam N
 * @param elements
 * @return Vector<T, N>
 */
template <typename T, size_t N>
RowVector<T, N> RowVec(std::initializer_list<T> elements) {
  std::array<T, N> temp_arr;
  size_t i = 0;
  for (T element : elements) {
    temp_arr[i++] = element;
  }

  return RowVector<T, N>(temp_arr);
}

/**
 * @brief Dot product of two column vectors
 *
 * @tparam T1
 * @tparam T2
 * @tparam N
 * @param vec1
 * @param vec2
 * @return decltype(std::declval<T1>() * std::declval<T2>())
 */
template <typename T1, typename T2, size_t N>
auto dot(Vector<T1, N> vec1, Vector<T2, N> vec2)
    -> decltype(std::declval<T1>() * std::declval<T2>()) {
  using T_out = decltype(std::declval<T1>() * std::declval<T2>());
  T_out dot_product = 0;
  for (size_t index = 0; index < N; index++) {
    dot_product += (vec1(0, index) * vec2(0, index));
  }
  return dot_product;
}

template <typename T1, size_t N> auto norm(Vector<T1, N> vec) {
  T1 square_sum = 0;

  for (size_t index = 0; index < N; index++) {
    square_sum += pow(vec(0, index), 2);
  }

  return sqrt(square_sum);
}

} // namespace tangent