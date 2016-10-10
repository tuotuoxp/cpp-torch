#pragma once
#include "../../include/nn/SpatialConvolutionMM.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::SpatialConvolutionMM<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    cpptorch::Tensor<TTensor> finput(true), fgradinput(true);
    cpptorch::Tensor<TTensor> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }

    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::NN<TTensor>::SpatialConvolutionMM_updateOutput(nullptr, input_new.valid() ? input_new : input, 
        out, weight_, bias_, finput, fgradinput, kW_, kH_, dW_, dH_, padW_, padH_);
    return out;
}
