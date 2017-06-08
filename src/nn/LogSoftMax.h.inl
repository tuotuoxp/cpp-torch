#pragma once
#include "../../include/nn/LogSoftMax.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::LogSoftMax<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{

    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::LogSoftMax_updateOutput(input, output);
    return output;
}
