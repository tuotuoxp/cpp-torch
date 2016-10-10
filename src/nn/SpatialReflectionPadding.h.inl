#pragma once
#include "../../include/nn/SpatialReflectionPadding.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::SpatialReflectionPadding<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    int idim = input.dim();
    assert((idim == 3 || idim == 4) && "input must be 3 or 4-dimensional");
    cpptorch::Tensor<TTensor> output(true);
    cpptorch::th::NN<TTensor>::SpatialReflectionPadding_updateOutput(nullptr, input, output, pad_l_, pad_r_, pad_t_, pad_b_);
    return output;
}
