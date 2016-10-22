#pragma once
#include "../../include/nn/SpatialReflectionPadding.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::SpatialReflectionPadding<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    int idim = input.dim();
    assert((idim == 3 || idim == 4) && "input must be 3 or 4-dimensional");
    cpptorch::Tensor<T,C> output(true);
    cpptorch::th::NN<T,C>::SpatialReflectionPadding_updateOutput(nullptr, input, output, pad_l_, pad_r_, pad_t_, pad_b_);
    return output;
}
