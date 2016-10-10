#pragma once
#include "../../include/nn/Inception.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Inception<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    if (input.dim() == 3)
    {
        return fromBatch(Decorator<TTensor>::forward(toBatch(input)));
    }
    else
    {
        return Decorator<TTensor>::forward(input);
    }
}
