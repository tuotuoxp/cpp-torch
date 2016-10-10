#pragma once
#include "../../include/nn/MulConstant.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::MulConstant<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    cpptorch::Tensor<TTensor> output(true);
    output.resizeAs(input);
    output.copy(input);
    output *= constant_scalar_;
    return output;
}
