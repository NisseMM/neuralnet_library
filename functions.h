#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <random>
#include <cmath>


namespace RANDOM
{
    extern size_t SEED;
    
    namespace BOUNDS
    {
        extern double UPPER;
        extern double LOWER;
    }
}

namespace FUNCTIONS
{
    template <typename T>
    T random()
    {
        static std::mt19937 gen(RANDOM::SEED); // Standard mersenne_twister_engine seeded with rd()
        static std::uniform_real_distribution<> dis(RANDOM::BOUNDS::LOWER, RANDOM::BOUNDS::UPPER);
        return dis(gen);
    }
    
    double relu(double value)
    {
        return std::max(0.0, value);
    }

    double clamp_one(double value)
    {
        return std::min(1.0, value);
    }
    
    double sigmoid(double value)
    {
        return 1 / (1 + exp(-value));
    }
    
    int sigmoid_i(double value)
    {
        return 1 / (1 + exp(-value));
    }

    double tanh(double value)
    {
        return std::tanh(value);
    }

}

#endif //FUNCTIONS_H