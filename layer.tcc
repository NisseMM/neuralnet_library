#include "layer.h"
#include "functions.h"
#include <iostream>

namespace NN
{
    template <typename T>
    Linnear<T>::Linnear(size_t in_features, size_t out_features)
        : W{Tensor<T>(in_features, out_features, FUNCTIONS::random)}, B{Tensor<T>(1, out_features)},
          X{Tensor<T>(0, 0)}, gradW{Tensor<T>(in_features, out_features)}, gradB{Tensor<T>(1, out_features)}
    {
    }
    
    
    template<typename T>
    Tensor<T> Linnear<T>::forward(Tensor<T> const& input)
    {
        X = input;
        Tensor<T> Y{X * W};
        return Y + B;
    }
    
    template<typename T>
    Tensor<T> Linnear<T>::backward(Tensor<T> const& grad)
    {
        auto XT = X.transpose();
        auto WT = W.transpose();

        gradW = XT * grad;
        gradB = grad.sum_rows();
    
        Tensor<T> gradX{grad * WT};    

        return gradX;
    }
    
    template<typename T>
    void Linnear<T>::step(double rate)
    {
    
        W = W - gradW * rate;
        B = B - gradB * rate;
    }

    template <typename T>
    Relu<T>::Relu()
        : Y{Tensor<T>(0, 0)}
    {
    }
    
    
    template<typename T>
    Tensor<T> Relu<T>::forward(Tensor<T> const& input)
    {
        Y = input;
        return Y.apply(FUNCTIONS::relu);
    }
    
    template<typename T>
    Tensor<T> Relu<T>::backward(Tensor<T> const& grad)
    {
        Y.apply(FUNCTIONS::clamp_one);
        return Y.element_mul(grad);
    }

    template <typename T>
    Sigmoid<T>::Sigmoid()
        : Y{Tensor<T>(0, 0)}
    {
    }
    
    
    template<typename T>
    Tensor<T> Sigmoid<T>::forward(Tensor<T> const& input)
    {
        Y = input;
        Y.apply(FUNCTIONS::sigmoid);
        return Y;
    }
    
    template<typename T>
    Tensor<T> Sigmoid<T>::backward(Tensor<T> const& grad)
    {
        return grad % (Y % (1.0 - Y));
    }

    template <typename T>
    Tanh<T>::Tanh()
        : Y{Tensor<T>(0, 0)}
    {
    }
    
    
    template<typename T>
    Tensor<T> Tanh<T>::forward(Tensor<T> const& input)
    {
        Y = input;
        Y.apply(FUNCTIONS::tanh);
        return Y;
    }
    
    template<typename T>
    Tensor<T> Tanh<T>::backward(Tensor<T> const& grad)
    {
        return grad % (1.0 - (Y % Y));
    }

}