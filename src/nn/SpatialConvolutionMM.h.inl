#pragma once
#include "../../include/nn/SpatialConvolutionMM.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::SpatialConvolutionMM<T>::forward(const cpptorch::Tensor<T> &input) const
{
    cpptorch::Tensor<T> finput(true), fgradinput(true);
    cpptorch::Tensor<T> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }

    cpptorch::Tensor<T> out(true);
    cpptorch::th::NN<T>::SpatialConvolutionMM_updateOutput(nullptr, input_new.valid() ? input_new : input, 
        out, weight_, bias_, finput, fgradinput, kW_, kH_, dW_, dH_, padW_, padH_);
    return out;
}
