#pragma once
#include "../../include/nn/SpatialMaxPooling.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::SpatialMaxPooling<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    cpptorch::Tensor<TTensor> output(true), indices(true);
    cpptorch::th::NN<TTensor>::SpatialMaxPooling_updateOutput(nullptr, input, output, indices, kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_);
    return output;
}
