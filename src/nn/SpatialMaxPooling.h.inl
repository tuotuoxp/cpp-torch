#pragma once
#include "../../include/nn/SpatialMaxPooling.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::SpatialMaxPooling<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    cpptorch::Tensor<T,C> output(true), indices(true);
    cpptorch::th::NN<T,C>::SpatialMaxPooling_updateOutput(input, output, indices, kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_);
    return output;
}
