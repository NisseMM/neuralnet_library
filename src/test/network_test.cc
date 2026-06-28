/*

    Test file for testing a basic neural network using the Network class

    Tests that a simple network can converge to XOR within a 0.01 margin in 10.000 iterations

*/

#include <cassert>
#include <cmath>

#include "functions.h"
#include "network.tcc"
#include "tensor.h"

double const DELTA_LIMIT{0.01};

namespace RANDOM
{
    std::random_device dev;
    size_t SEED = dev();
    namespace BOUNDS
    {
        double UPPER = 0.7;
        double LOWER = -0.7;
    }
}   

int main()
{
    Tensor<double> X{
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1}};

    std::vector<double> vals{0, 1, 1, 0};
    Tensor<double> Y = Tensor<double>(4, 1, vals);

    size_t MAXITERS{10*1000};

    Network net(10,
                LAYER::Linnear(2, 4),
                LAYER::Tanh(),
                LAYER::Linnear(4, 1),
                LAYER::Sigmoid());

    net.train(MAXITERS, X, Y);

    auto pred = net.predict(X);

    for (size_t i{}; i < Y.shape().second; ++i)
        assert(std::abs(pred.at(0, i) - Y.at(0, i)) < DELTA_LIMIT);

    return 0;
}