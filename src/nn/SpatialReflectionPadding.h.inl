#pragma once
#include "../../include/nn/SpatialReflectionPadding.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::SpatialReflectionPadding<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    int idim = input.dim();
    assert((idim == 3 || idim == 4) && "input must be 3 or 4-dimensional");
    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::SpatialReflectionPadding_updateOutput(input, output, pad_l_, pad_r_, pad_t_, pad_b_);
    return output;
}
