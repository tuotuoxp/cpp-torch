#pragma once
#include "../../include/nn/SpatialMaxPooling.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::SpatialMaxPooling<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> output(true), indices(true);
    cpptorch::th::NN<T, F>::SpatialMaxPooling_updateOutput(input, output, indices, kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_);
    return output;
}
