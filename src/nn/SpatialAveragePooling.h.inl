#pragma once
#include "../../include/nn/SpatialAveragePooling.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::SpatialAveragePooling<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    cpptorch::Tensor<TTensor> output(true);
    cpptorch::th::NN<TTensor>::SpatialAveragePooling_updateOutput(nullptr, input, output,
        kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_, count_include_pad_);

    // for backward compatibility with saved models which are not supposed to have "divide" field
    if (!divide_)
    {
        output *= (typename TTensor::Storage::Base)(kW_ * kH_);
    }
    return output;
}
