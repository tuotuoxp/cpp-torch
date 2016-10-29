#pragma once
#include "../../include/nn/Sqrt.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Sqrt<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::Sqrt_updateOutput(input, output, eps_);
    return output;
}
