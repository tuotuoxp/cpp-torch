#pragma once
#include "../../include/nn/SpatialReflectionPadding.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::SpatialReflectionPadding<T>::forward(const cpptorch::Tensor<T> &input) const
{
    int idim = input.dim();
    assert((idim == 3 || idim == 4) && "input must be 3 or 4-dimensional");
    cpptorch::Tensor<T> output(true);
    cpptorch::th::NN<T>::SpatialReflectionPadding_updateOutput(nullptr, input, output, pad_l_, pad_r_, pad_t_, pad_b_);
    return output;
}
