#include <array>
#include <iostream>
#include <math.h>

#pragma once

namespace tangent {
template <typename T, size_t rows, size_t columns> class Matrix {
public:
  Matrix(const std::array<T, rows * columns> &input_arr) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        container[i][j] = input_arr[i * columns + j];
      }
    }
  }

  Matrix() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        container[i][j] = 0;
      }
    }
  }

  /**
   * @brief Allows accessing particular elements in a matrix using row-col
   * protocol (e.g. matrix(2, 3))
   *
   * @param row
   * @param col
   * @return T
   */
  T &operator()(size_t row, size_t col) { return container[row][col]; }

  /**
   * @brief Const overload for () operator
   *
   * @param row
   * @param col
   * @return T
   */
  const T &operator()(size_t row, size_t col) const {
    return container[row][col];
  }

  /**
   * @brief Returns a sum of two matrices - expects same type so will throw an
   * error if there is a type mismatch
   *
   * @param rhs
   * @return Matrix&
   */
  Matrix operator+(const Matrix &rhs) const {
    Matrix output; // same T, rows, cols
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < columns; ++col) {
        output(row, col) = (*this)(row, col) + rhs(row, col);
      }
    }
    return output;
  }

  /**
   * @brief Returns a product of two matrices - checks for dimensional
   * compatibility and then computes matrix product
   *
   * @param rhs
   * @return Matrix&
   */
  template <typename T_rhs, size_t rows_rhs, size_t cols_rhs>
  auto operator*(const Matrix<T_rhs, rows_rhs, cols_rhs> &rhs) const {
    static_assert(
        columns == rows_rhs,
        "Matrix multiplication cannot be done due to incompatible dimensions");
    Matrix<T, rows, cols_rhs> output;
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols_rhs; col++) {
        T value = 0;
        for (size_t col_lhs = 0; col_lhs < columns; col_lhs++) {
          output(row, col) += (*this)(row, col_lhs) * rhs(col_lhs, col);
        }
      }
    }
    return output;
  }

  void print() {
    // Print in beautified form, row by row
    for (size_t row_num = 0; row_num < rows; row_num++) {
      for (size_t col_num = 0; col_num < columns; col_num++) {
        std::cout << container[row_num][col_num] << " ";
      }
      std::cout << std::endl;
    }
  }

private:
  T container[rows][columns];
};
} // namespace tangent