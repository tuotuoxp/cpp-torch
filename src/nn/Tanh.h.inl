#pragma once
#include "../../include/nn/Tanh.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Tanh<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{

    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::Tanh_updateOutput(input, output);
    return output;
}
