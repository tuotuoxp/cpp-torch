#pragma once
#include "../../include/nn/SpatialConvolution.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::SpatialConvolution<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    cpptorch::Tensor<T,C> finput(true), fgradinput(true);
    cpptorch::Tensor<T,C> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }

    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::NN<T,C>::SpatialConvolutionMM_updateOutput(nullptr, input_new.valid() ? input_new : input, 
        out, weight_, bias_, finput, fgradinput, kW_, kH_, dW_, dH_, padW_, padH_);
    return out;
}
