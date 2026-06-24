#include <format>
#include <iostream>

#include "functions.h"
#include "network.tcc"
#include "tensor.h"

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

    Tensor<double> Y = Tensor<double>{1, 0, 0, 1}.transpose();

    size_t MAXITERS{1000 * 1000};

    

    Network net(10,
                LAYER::Linnear(2, 4),
                LAYER::Tanh(),
                LAYER::Linnear(4, 1),
                LAYER::Sigmoid());

    net.train(MAXITERS, X, Y);

    std::cout << std::format("Result\n{}", net.predict(X)) << std::endl;

    return 0;
}