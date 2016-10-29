#pragma once
#include "../../include/nn/Square.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Square<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::Square_updateOutput(input, output);
    return output;
}
