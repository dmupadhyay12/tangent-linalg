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
 * @brief
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

template <typename T, size_t N>
RowVector<T, N> RowVec(std::initializer_list<T> elements) {
  std::array<T, N> temp_arr;
  size_t i = 0;
  for (T element : elements) {
    temp_arr[i++] = element;
  }

  return RowVector<T, N>(temp_arr);
}

} // namespace tangent