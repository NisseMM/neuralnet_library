#ifndef MATRIX_TCC
#define MATRIX_TCC

#include "matrix.h"
#include <cstddef>
#include <initializer_list>

template <typename T, size_t ROWS, size_t COLS>
Matrix<T, ROWS, COLS>::Matrix()
    : values{T{}}, transposed{false}
{
}

template <typename T, size_t ROWS, size_t COLS>
Matrix<T, ROWS, COLS>::Matrix(T default_val)
    : Matrix{}
{
    for (size_t row{}; row < ROWS; ++row)
        for (size_t col{}; col < COLS; ++col)
            at(row, col) = default_val;
}

template <typename T, size_t ROWS, size_t COLS>
Matrix<T, ROWS, COLS>::Matrix(std::initializer_list<T> list)
    : Matrix{}
{
    assert(list.size() == ROWS * COLS);
    size_t index{};
    for (auto && val : list)
        values[index++] = val;
}

template <typename T, size_t ROWS, size_t COLS>
T const &Matrix<T, ROWS, COLS>::at(size_t row, size_t col) const noexcept
{
    return transposed ? values[col * COLS + row] : values[row * COLS + col];
}

template <typename T, size_t ROWS, size_t COLS>
T &Matrix<T, ROWS, COLS>::at(size_t row, size_t col) noexcept
{
    return const_cast<T &>(static_cast<Matrix const *>(this)->at(row, col));
}

template <typename T, size_t ROWS, size_t COLS>
void Matrix<T, ROWS, COLS>::transpose() noexcept
{
    transposed = not transposed;
}

template <typename T, size_t ROWS, size_t COLS>
std::pair<size_t, size_t> Matrix<T, ROWS, COLS>::shape() const noexcept
{
    return transposed ? std::make_pair(COLS, ROWS) : std::make_pair(ROWS, COLS);
}

template <typename T, size_t ROWS, size_t COLS>
Matrix<T, ROWS, COLS> &Matrix<T, ROWS, COLS>::operator+=(Matrix<T, ROWS, COLS> const &other)
{
    for (size_t row{}; row < ROWS; ++row)
        for (size_t col{}; col < COLS; ++col)
            at(row, col) += other.at(row, col);

    return *this;
}

template <typename T, size_t ROWS, size_t COLS>
Matrix<T, ROWS, COLS> &Matrix<T, ROWS, COLS>::operator-=(Matrix<T, ROWS, COLS> const &other)
{
    for (size_t row{}; row < ROWS; ++row)
        for (size_t col{}; col < COLS; ++col)
            at(row, col) -= other.at(row, col);

    return *this;
}

template <typename T, size_t ROWS, size_t COLS>
Matrix<T, ROWS, COLS> Matrix<T, ROWS, COLS>::operator+(Matrix<T, ROWS, COLS> const &other) const
{
    Matrix tmp{*this};
    return tmp += other;
}

template <typename T, size_t ROWS, size_t COLS>
Matrix<T, ROWS, COLS> Matrix<T, ROWS, COLS>::operator-(Matrix<T, ROWS, COLS> const &other) const
{
    Matrix tmp{*this};
    return tmp -= other;
}

template <typename T, size_t ROWS, size_t COLS>
template <size_t O_ROWS, size_t O_COLS>
Matrix<T, ROWS, O_COLS> Matrix<T, ROWS, COLS>::operator*(Matrix<T, O_ROWS, O_COLS> const &other) const
{
    return naive_multiplication(other);
}


template <typename T, size_t ROWS, size_t COLS>
template <size_t O_ROWS, size_t O_COLS>
Matrix<T, ROWS, O_COLS> Matrix<T, ROWS, COLS>::naive_multiplication(Matrix<T, O_ROWS, O_COLS> const &other) const
{
    static_assert(COLS == O_ROWS);
    Matrix<T, ROWS, O_COLS> res{};
    for (size_t row{}; row < ROWS; ++row)
        for (size_t o_col{}; o_col < O_COLS; ++o_col)
            for (size_t col{}; col < COLS; ++col)
                res.at(row, o_col) += at(row, col) * other.at(col, o_col);
    return res;
}


#endif