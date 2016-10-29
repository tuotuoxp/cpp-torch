#pragma once
#include "../../include/nn/SpatialAveragePooling.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::SpatialAveragePooling<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::SpatialAveragePooling_updateOutput(input, output,
        kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_, count_include_pad_);

    // for backward compatibility with saved models which are not supposed to have "divide" field
    if (!divide_)
    {
        output *= (T)(kW_ * kH_);
    }
    return output;
}
