#include "include/matrix.hpp"

int main() {
  tangent::Matrix<int, 1, 3> matrix_1 = {{2, 1, 3}};
  tangent::Matrix<int, 3, 2> matrix_2 = {{1, 2, 3, 5, 4, 6}};
  auto matrix_3 = matrix_1 * matrix_2;

  matrix_3.print();
}