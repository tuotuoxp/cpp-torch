#pragma once
#include "../../include/nn/SpatialConvolution.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::SpatialConvolution<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> finput(true), fgradinput(true);
    cpptorch::Tensor<T, F> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }

    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::NN<T, F>::SpatialConvolutionMM_updateOutput(input_new.valid() ? input_new : input, 
        out, weight_, bias_, finput, fgradinput, kW_, kH_, dW_, dH_, padW_, padH_);
    return out;
}
