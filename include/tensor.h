#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <utility>
#include <vector>
#include <memory>
#include <format>
#include <string>

template <typename T = double>
class Tensor
{
public:
    using storage_type = std::vector<T>;

    Tensor(size_t, size_t = 1, T const & = T{});
    Tensor(std::initializer_list<T>);
    Tensor(std::initializer_list<std::initializer_list<T>>);
    Tensor(size_t, size_t, T (*)());
    Tensor(size_t, size_t, std::vector<T> &);

    T &at(size_t, size_t) noexcept;
    T const &at(size_t, size_t) const noexcept;

    std::pair<size_t, size_t> shape() const;

    Tensor transpose() const;

    template <typename Func>
    Tensor &apply(Func);

    Tensor &negate();

    template <typename B>
    auto operator*(B) const;

    template <typename B>
    auto operator*(Tensor<B> const &other) const;

    template <typename B>
    auto operator%(Tensor<B> const &other) const;

    template <typename B>
    auto operator+(B val) const;

    template <typename B>
    auto operator+(Tensor<B> const &other) const;

    template <typename B>
    auto operator-(B val) const;

    template <typename B>
    auto operator-(Tensor<B> const &other) const;

    Tensor sum_rows() const;

    template <typename B>
    auto element_mul(Tensor<B> const &other) const;

    std::string to_string() const;

private:
    size_t rows;
    size_t cols;
    size_t row_stride;
    size_t col_stride;
    storage_type data;

    template <typename B>
    auto naive_multiplication(Tensor<B> const &other) const;
};

template <typename T, typename B>
Tensor<T> operator-(T lhs, Tensor<T> const &rhs);

template <typename T>
struct std::formatter<Tensor<T>> : std::formatter<std::string_view>
{
    auto format(Tensor<T> const &tensor, format_context &context) const
    {
        return std::formatter<std::string_view>::format(tensor.to_string(), context);
    }
};

#include "tensor.tcc"
#endif