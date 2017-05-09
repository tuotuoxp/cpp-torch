#pragma once
#include "../../include/nn/SoftMax.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::SoftMax<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{

    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::SoftMax_updateOutput(input, output);
    return output;
}
