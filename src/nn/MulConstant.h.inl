#pragma once
#include "../../include/nn/MulConstant.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::MulConstant<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    cpptorch::Tensor<T,C> output(true);
    output.resizeAs(input);
    output.copy(input);
    output *= constant_scalar_;
    return output;
}
