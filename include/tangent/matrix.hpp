#include <array>
#include <iostream>
#include <math.h>

#pragma once

namespace tangent {
template <typename T, size_t rows, size_t columns> class Matrix {
public:
  Matrix(const std::array<T, rows * columns> &input_arr) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        container[i][j] = input_arr[i * columns + j];
      }
    }
  }

  Matrix() {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
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
        for (size_t col_lhs = 0; col_lhs < columns; col_lhs++) {
          output(row, col) += (*this)(row, col_lhs) * rhs(col_lhs, col);
        }
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
  auto operator^(const Matrix<T_rhs, rows_rhs, cols_rhs> &rhs) const {
    static_assert(
        columns == rows_rhs,
        "Matrix multiplication cannot be done due to incompatible dimensions");
    Matrix<T, rows, cols_rhs> output;
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols_rhs; col++) {
        for (size_t col_lhs = 0; col_lhs < columns; col_lhs++) {
          output(row, col) += (*this)(row, col_lhs) * rhs(col_lhs, col);
        }
      }
    }
    return output;
  }

  /**
   * @brief Returns the scalar multiplication of a matrix and a constant -
   * checks for dimensional compatibility and then computes matrix product
   *
   * @param scalar_multiplier
   * @return Matrix&
   */
  template <class Scalar>
  requires std::is_arithmetic_v<Scalar>
  auto operator*(const Scalar scalar_multiplier) const
      -> Matrix<decltype(std::declval<T>() * std::declval<Scalar>()), rows,
                columns> {
    using outType = decltype(std::declval<T>() * std::declval<Scalar>());
    Matrix<outType, rows, columns> output;
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < columns; col++) {
        output(row, col) = (*this)(row, col) * scalar_multiplier;
      }
    }
    return output;
  }

  /**
   * @brief Returns whether two matrices are equal to one another
   *
   * @param to_check_equality_of
   * @return Whether the two matrices are equal to one another or not
   */

  template <typename T_rhs, size_t rows_rhs, size_t cols_rhs>
  bool operator==(const Matrix<T_rhs, rows_rhs, cols_rhs> &rhs) {
    if (rows_rhs != rows || cols_rhs != columns) {
      return false;
    } else {
      for (size_t row = 0; row < rows_rhs; row++) {
        for (size_t col = 0; col < cols_rhs; col++) {
          if ((*this)(row, col) != rhs(row, col)) {
            return false;
          }
        }
      }
    }
    return true;
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

/**
 * @brief Returns the scalar multiplication of a matrix and a constant -
 * checks for dimensional compatibility and then computes matrix product
 *
 * @param scalar_multiplier
 * @return Matrix&
 */
template <class Scalar, class T, size_t rows, size_t cols>
requires std::is_arithmetic_v<Scalar>
auto operator*(Scalar scalar_multiplier, Matrix<T, rows, cols> matrix)
    -> Matrix<decltype(std::declval<T>() * std::declval<Scalar>()), rows,
              cols> {
  using outType = decltype(std::declval<T>() * std::declval<Scalar>());
  Matrix<outType, rows, cols> output;
  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      output(row, col) = matrix(row, col) * scalar_multiplier;
    }
  }
  return output;
}
} // namespace tangent