#pragma once
#include "../../include/nn/SpatialAveragePooling.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::SpatialAveragePooling<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    cpptorch::Tensor<T,C> output(true);
    cpptorch::th::NN<T,C>::SpatialAveragePooling_updateOutput(nullptr, input, output,
        kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_, count_include_pad_);

    // for backward compatibility with saved models which are not supposed to have "divide" field
    if (!divide_)
    {
        output *= (T)(kW_ * kH_);
    }
    return output;
}
