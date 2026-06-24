#include <cassert>
#include <iostream>
#include <utility>

#include "matrix.h"

int main()
{
    Matrix<double, 10, 10> m{};

    assert(0.0 == m.at(5, 5));
    assert(0.0 == m.at(8, 9));

    m.at(1, 1) = 30.5;

    assert(m.at(1, 1) == 30.5);

    Matrix<double, 10, 10> const n(12.5);

    assert(12.5 == n.at(5, 5));
    assert(12.5 == n.at(8, 9));

    Matrix<double, 10, 10> const b(10.5);

    Matrix res{n + b};

    assert(12.5 == n.at(5, 5));
    assert(12.5 == n.at(8, 9));

    assert(10.5 == b.at(5, 5));
    assert(10.5 == b.at(8, 9));

    assert(23.0 == res.at(5, 5));
    assert(23.0 == res.at(8, 9));

    res = n - b;

    assert(12.5 == n.at(5, 5));
    assert(12.5 == n.at(8, 9));

    assert(10.5 == b.at(5, 5));
    assert(10.5 == b.at(8, 9));

    assert(2.0 == res.at(5, 5));
    assert(2.0 == res.at(8, 9));

    Matrix<double, 2, 2> A{3, 7, 4, 5};
    Matrix<double, 2, 1> B{2, 9};

    assert(3 == A.at(0, 0));
    assert(7 == A.at(0, 1));
    assert(4 == A.at(1, 0));
    assert(5 == A.at(1, 1));

    assert(2 == B.at(0, 0));
    assert(9 == B.at(1, 0));

    Matrix mul{A * B};

    A.transpose();

    assert(3 == A.at(0, 0));
    assert(7 == A.at(1, 0));
    assert(4 == A.at(0, 1));
    assert(5 == A.at(1, 1));

    Matrix<double, 2, 3> C{3, 7, 4, 5, 6, 10};

    assert(C.shape() == std::make_pair(size_t{2}, size_t{3}));
    assert(3 == C.at(0, 0));
    assert(7 == C.at(0, 1));
    assert(4 == C.at(0, 2));
    assert(5 == C.at(1, 0));
    assert(6 == C.at(1, 1));
    assert(10 == C.at(1, 2));

    C.transpose();

    assert(C.shape() == std::make_pair(size_t{3}, size_t{2}));

    assert(3 == C.at(0, 0));
    assert(5 == C.at(0, 1));
    assert(7 == C.at(1, 0));
    assert(6 == C.at(1, 1));
    assert(4 == C.at(2, 0));
    assert(10 == C.at(2, 1));

    return 0;
}