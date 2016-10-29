#pragma once
#include "../../include/nn/MulConstant.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::MulConstant<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> output(true);
    output.resizeAs(input);
    output.copy(input);
    output *= constant_scalar_;
    return output;
}
