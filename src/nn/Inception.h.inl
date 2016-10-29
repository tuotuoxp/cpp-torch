#pragma once
#include "../../include/nn/Inception.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Inception<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    if (input.dim() == 3)
    {
        return fromBatch(Decorator<T, F>::forward(toBatch(input)));
    }
    else
    {
        return Decorator<T, F>::forward(input);
    }
}
