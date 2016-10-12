#pragma once
#include "../../include/nn/SpatialAveragePooling.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::SpatialAveragePooling<T>::forward(const cpptorch::Tensor<T> &input) const
{
    cpptorch::Tensor<T> output(true);
    cpptorch::th::NN<T>::SpatialAveragePooling_updateOutput(nullptr, input, output,
        kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_, count_include_pad_);

    // for backward compatibility with saved models which are not supposed to have "divide" field
    if (!divide_)
    {
        output *= (T)(kW_ * kH_);
    }
    return output;
}
