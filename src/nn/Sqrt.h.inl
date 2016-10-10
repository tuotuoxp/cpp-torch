#pragma once
#include "../../include/nn/Sqrt.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Sqrt<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    cpptorch::Tensor<TTensor> output(true);
    cpptorch::th::NN<TTensor>::Sqrt_updateOutput(nullptr, input, output, eps_);
    return output;
}
