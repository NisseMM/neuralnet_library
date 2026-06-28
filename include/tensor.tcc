#include "tensor.h"

#include <cstddef>
#include <utility>
#include <memory>
#include <algorithm>
#include <sstream>
#include <string>
#include <cassert>

#include <iostream>

template <typename T>
Tensor<T>::Tensor(size_t rows, size_t cols, T const &initial)
    : rows{rows}, cols{cols}, row_stride{cols}, col_stride{1}, data{storage_type(rows * cols, initial)}
{
}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<T> list)
    : rows{1}, cols{list.size()}, row_stride{cols}, col_stride{1}, data{storage_type()}
{
    assert(list.size() == rows * cols);
    std::copy(begin(list), end(list), std::back_inserter((data)));
}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<std::initializer_list<T>> list)
    : rows{list.size()}, cols{list.begin()->size()}, row_stride{cols}, col_stride{1}, data{storage_type()}
{
    assert(list.size() == rows);
    std::for_each(begin(list), end(list), [this](auto l)
                  { 
                    assert(l.size() == this->cols); 
                    std::copy(begin(l), end(l), std::back_inserter(this->data)); });
}

template <typename T>
Tensor<T>::Tensor(size_t rows, size_t cols, T (*func)())
    : Tensor(rows, cols, T{})
{
    // assert(list.size() == (data).size());
    std::generate(begin(data), end(data), func);
}

template <typename T>
Tensor<T>::Tensor(size_t rows, size_t cols, std::vector<T> & init)
    : rows{rows}, cols{cols}, row_stride{cols}, col_stride{1}, data{}
{
    std::swap(data, init);
}

template <typename T>
T &Tensor<T>::at(size_t row, size_t col) noexcept
{
    return const_cast<T &>(static_cast<Tensor const *>(this)->at(row, col));
}

template <typename T>
T const &Tensor<T>::at(size_t row, size_t col) const noexcept
{
    return (data)[row * row_stride + col * col_stride];
}

template <typename T>
std::pair<size_t, size_t> Tensor<T>::shape() const
{
    return {rows, cols};
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const
{
    Tensor<T> out(cols, rows); // swap dimensions

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out.at(j, i) = at(i, j);

    return out;
}

template <typename T>
template <typename Func>
Tensor<T> &Tensor<T>::apply(Func func)
{
    using R = std::invoke_result_t<Func, T>;
    static_assert(std::is_same_v<R, T>, "Func must return T");

    std::transform(begin(data), end(data), begin(data), func);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::negate()
{
    return apply([](T val)
                 { return val * -1; });
}

template <typename T>
template <typename B>
auto Tensor<T>::operator*(B val) const
{
    using R = std::common_type_t<T, B>;
    Tensor<R> ret{*this};
    return ret.apply([val](R curr)
                     { return curr * val; });
}

template <typename T>
template <typename B>
auto Tensor<T>::operator*(Tensor<B> const &other) const
{
    assert(cols == other.rows);
    return naive_multiplication(other);
}

template <typename T>
template <typename B>
auto Tensor<T>::operator%(Tensor<B> const &other) const
{
    assert(cols == other.cols and rows == other.rows);
    using R = std::common_type_t<T, B>;
    Tensor<R> ret{*this};
    std::transform(begin(ret.data), end(ret.data), begin(other.data), begin(ret.data),
                   [](auto a, auto b)
                   { return a * b; });
    return ret;
}

template <typename T>
template <typename B>
auto Tensor<T>::operator+(B val) const
{
    using R = std::common_type_t<T, B>;
    Tensor<R> ret{*this};
    return ret.apply([val](R curr)
                     { return curr + val; });
}

template <typename T>
template <typename B>
auto Tensor<T>::operator+(Tensor<B> const &other) const
{
    using R = std::common_type_t<T, B>;
    Tensor<R> out(rows, cols);

    if (other.rows == 1 && other.cols == cols)
    {
        // row-wise broadcast
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                out.at(r, c) = at(r, c) + other.at(0, c);
        return out;
    }

    if (other.cols == 1 && other.rows == rows)
    {
        // column-wise broadcast
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                out.at(r, c) = at(r, c) + other.at(r, 0);
        return out;
    }

    assert(other.rows == rows and other.cols == cols);
    for (size_t row{}; row < out.rows; ++row)
        for (size_t col{}; col < out.cols; ++col)
            out.at(row, col) = at(row, col) + other.at(row, col);
    return out;
}

template <typename T>
template <typename B>
auto Tensor<T>::operator-(B val) const
{
    using R = std::common_type_t<T, B>;
    Tensor<R> ret(*this);

    return ret.apply([val](R curr)
                     { return curr - val; });
}

template <typename T>
template <typename B>
auto Tensor<T>::operator-(Tensor<B> const &other) const
{
    using R = std::common_type_t<T, B>;
    Tensor<R> out(rows, cols);

    if (other.rows == 1 and rows != 1)
    {
        assert(other.cols == cols);
        for (size_t row{}; row < out.rows; ++row)
            for (size_t col{}; col < out.cols; ++col)
                out.at(row, col) = at(row, col) - other.at(0, col);
        return out;
    }

    assert(other.rows == rows and other.cols == cols);
    for (size_t row{}; row < out.rows; ++row)
        for (size_t col{}; col < out.cols; ++col)
            out.at(row, col) = at(row, col) - other.at(row, col);
    return out;
}

template <typename T>
template <typename B>
auto Tensor<T>::naive_multiplication(Tensor<B> const &other) const
{
    using R = std::common_type_t<T, B>;
    Tensor<R> res(rows, other.cols);
    for (size_t row{}; row < rows; ++row)
    {
        for (size_t col{}; col < cols; ++col)
        {
            auto const &a{at(row, col)};
            for (size_t o_col{}; o_col < other.cols; ++o_col)
                res.at(row, o_col) += a * other.at(col, o_col);
        }
    }
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::sum_rows() const
{
    Tensor<T> out(1, cols, T{});
    for (size_t j{}; j < cols; ++j)
        for (size_t i{}; i < rows; ++i)
            out.at(0, j) += at(i, j);
    return out;
}

template <typename T>
template <typename B>
auto Tensor<T>::element_mul(Tensor<B> const &other) const
{
    using R = std::common_type_t<T, B>;
    Tensor<R> res(rows, cols);
    std::transform(begin(data), end(data), begin(other.data), begin(res.data), [](auto a, auto b)
                   { return a * b; });
    return res;
}

template <typename T>
std::string Tensor<T>::to_string() const
{
    std::ostringstream oss{};
    for (size_t row{}; row < rows; ++row)
    {
        for (size_t col{}; col < cols; ++col)
        {
            oss << at(row, col);
            if (col != cols - 1)
                oss << " ";
        }
        if (row != rows - 1)
            oss << "\n";
    }
    return oss.str();
}

template <typename T>
Tensor<T> operator-(T lhs, Tensor<T> const &rhs)
{
    Tensor<T> ret{rhs};
    return ret.apply([lhs](T curr)
                     { return lhs - curr; });
}