#pragma once
#include "../../include/nn/SpatialMaxPooling.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::SpatialMaxPooling<T>::forward(const cpptorch::Tensor<T> &input) const
{
    cpptorch::Tensor<T> output(true), indices(true);
    cpptorch::th::NN<T>::SpatialMaxPooling_updateOutput(nullptr, input, output, indices, kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_);
    return output;
}
