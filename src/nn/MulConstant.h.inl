#pragma once
#include "../../include/nn/MulConstant.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::MulConstant<T>::forward(const cpptorch::Tensor<T> &input) const
{
    cpptorch::Tensor<T> output(true);
    output.resizeAs(input);
    output.copy(input);
    output *= constant_scalar_;
    return output;
}
