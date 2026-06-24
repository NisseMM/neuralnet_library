#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <initializer_list>

template <typename T, size_t ROWS, size_t COLS>
class Matrix
{
public:
  Matrix();
  Matrix(T);
  Matrix(std::initializer_list<T>);

  T const& at(size_t, size_t) const noexcept;
  T & at(size_t, size_t) noexcept;

  void transpose() noexcept;
  std::pair<size_t, size_t> shape() const noexcept;

  Matrix& operator+=(Matrix const&);
  Matrix& operator-=(Matrix const&);
  Matrix operator+(Matrix const&) const;
  Matrix operator-(Matrix const&) const;

  template <size_t O_ROWS, size_t O_COLS>
  Matrix<T, ROWS, O_COLS> operator*(Matrix<T, O_ROWS, O_COLS> const &) const;

private:
  T values[ROWS * COLS];
  bool transposed;

  template <size_t O_ROWS, size_t O_COLS>
  Matrix<T, ROWS, O_COLS> naive_multiplication(Matrix<T, O_ROWS, O_COLS> const &other) const;
};

#include "matrix.tcc"
#endif