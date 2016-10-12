#pragma once
#include "../../include/nn/Inception.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::Inception<T>::forward(const cpptorch::Tensor<T> &input) const
{
    if (input.dim() == 3)
    {
        return fromBatch(Decorator<T>::forward(toBatch(input)));
    }
    else
    {
        return Decorator<T>::forward(input);
    }
}
