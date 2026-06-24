#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

namespace NN
{
    template <typename T=double>
    class Layer
    {
    public:
        virtual Tensor<T> forward(Tensor<T> const &input) = 0;
        virtual Tensor<T> backward(Tensor<T> const &gradient) = 0;
    
        virtual void step(double) {}
        // virtual void zero_grad() {}
    };
    
    template <typename T=double>
    class Linnear : public Layer<T>
    {
    public:
        Linnear(size_t in_features, size_t out_features);
    
        Tensor<T> forward(Tensor<T> const &input) override;
        Tensor<T> backward(Tensor<T> const &gradient) override;
    
        void step(double rate) override;
        // void zero_grad() override;
    
    private:
        Tensor<T> W;
        Tensor<T> B;
    
        Tensor<T> X;
    
        Tensor<T> gradW;
        Tensor<T> gradB;
    };
    
    template <typename T=double>
    class Relu : public Layer<T>
    {
    public:
        Relu();
    
        Tensor<T> forward(Tensor<T> const &input) override;
        Tensor<T> backward(Tensor<T> const &gradient) override;
    
    private:
        Tensor<T> Y;
    };

    template <typename T=double>
    class Sigmoid : public Layer<T>
    {
    public:
        Sigmoid();
    
        Tensor<T> forward(Tensor<T> const &input) override;
        Tensor<T> backward(Tensor<T> const &gradient) override;
    
    private:
        Tensor<T> Y;
    };

    template <typename T=double>
    class Tanh : public Layer<T>
    {
    public:
        Tanh();
    
        Tensor<T> forward(Tensor<T> const &input) override;
        Tensor<T> backward(Tensor<T> const &gradient) override;
    
    private:
        Tensor<T> Y;
    };

    template<typename T>
    Tensor<T> dMSE(Tensor<T> const& real, Tensor<T> const& pred)
    {
        assert(real.shape() == pred.shape());
        return  (pred - real) * (2.0/(real.shape().first * real.shape().second));
    }

    template<typename T>
    Tensor<T> dBCE(Tensor<T> const& real, Tensor<T> const& pred)
    {
        assert(real.shape() == pred.shape());
        return  (pred - real) * (1.0 / real.shape().first);
    }

    template<typename T>
    Tensor<T> BCE(Tensor<T> const& real, Tensor<T> const& pred)
    {
        assert(real.shape() == pred.shape());

        const double eps = 1e-12; // avoid log(0)
        size_t N = real.shape().first; // batch size

        T loss_value = 0;

        for (size_t i = 0; i < N; ++i)
        {
            double y = real.at(i, 0);
            double p = pred.at(i, 0);

            // clamp to avoid log(0)
            p = std::clamp(p, eps, 1.0 - eps);

            loss_value += -(y * std::log(p) + (1 - y) * std::log(1 - p));
        }

        loss_value /= N;

        // return as a 1×1 tensor
        return Tensor<T>(1, 1, loss_value);
    }

}

namespace LAYER
{
    template<typename T=double, typename ...Args>
    std::unique_ptr<NN::Layer<T>> Linnear(size_t in, size_t out, Args...list)
    {
        return std::make_unique<NN::Linnear<T>>(in, out, list...);
    }

    template<typename T=double>
    std::unique_ptr<NN::Layer<T>> Relu()
    {
        return std::make_unique<NN::Relu<T>>();
    }

    template<typename T=double>
    std::unique_ptr<NN::Layer<T>> Sigmoid()
    {
        return std::make_unique<NN::Sigmoid<T>>();
    }
    
    template<typename T=double>
    std::unique_ptr<NN::Layer<T>> Tanh()
    {
        return std::make_unique<NN::Tanh<T>>();
    }
}


#include "layer.tcc"
#endif // LAYER_H