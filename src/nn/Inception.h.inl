#pragma once
#include "../../include/nn/Inception.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::Inception<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    if (input.dim() == 3)
    {
        return fromBatch(Decorator<T,C>::forward(toBatch(input)));
    }
    else
    {
        return Decorator<T,C>::forward(input);
    }
}
