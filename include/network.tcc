#ifndef NETWORK_H
#define NETWORK_H

#include "functions.h"

#include "tensor.h"
#include "layer.h"

#include <vector>
#include <algorithm>
#include <ranges>
#include <memory>

template<typename T=double>
class Network
{
public:
    template<typename ...Args>
    Network(double rate,  Args&&... list)
        : lr{rate}, layers{}
    {
        (layers.emplace_back(std::forward<Args>(list)), ...);
    }

    Tensor<T> forward(Tensor<T> const& X)
    {
        Tensor<T> prev{X};

        for (auto const& layer : layers)
        {
            // std::cout << typeid(*layer).name() << std::endl;
            prev = layer->forward(prev);
        }
        return prev;
    }

    void backward(Tensor<T> const& loss)
    {
        Tensor<T> prev{loss};
        for (auto const& layer : layers | std::views::reverse)
        {
            // std::cout << typeid(*layer).name() << std::endl;
            prev = layer->backward(prev);
        }
    }

    void step()
    {
        for (auto const& layer : layers)
        {
            layer->step(lr);
        }
    }

    Tensor<T> predict(Tensor<T> const& X)
    {
        return forward(X);
    }

    void train(size_t iterations, Tensor<T> const& X, Tensor<T> const& Y)
    {
        for (size_t i{}; i < iterations; ++i)
        {
            // std::cout << std::format("\n\nIteration i={}\n", i);
            auto pred{forward(X)};
            auto grad{NN::dBCE(Y, pred)};
            // if (i % 100 == 0)
            // {
            //     std::cout << std::format("Iter\n{}", i) << std::endl;
            //     std::cout << std::format("Loss\n{}\n\n", NN::BCE(Y, pred)) << std::endl;
            // }
            backward(grad);
            // std::cout << "\n" << std::endl;
            step();
        }
    }
    
private:
    double lr;
    std::vector<std::unique_ptr<NN::Layer<T>>> layers;
    
};

#endif //NETWORK_H