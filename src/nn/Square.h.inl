#pragma once
#include "../../include/nn/Square.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Square<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    cpptorch::Tensor<TTensor> output(true);
    cpptorch::th::NN<TTensor>::Square_updateOutput(nullptr, input, output);
    return output;
}
