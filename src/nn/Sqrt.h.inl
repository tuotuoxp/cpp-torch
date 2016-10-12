#pragma once
#include "../../include/nn/Sqrt.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::Sqrt<T>::forward(const cpptorch::Tensor<T> &input) const
{
    cpptorch::Tensor<T> output(true);
    cpptorch::th::NN<T>::Sqrt_updateOutput(nullptr, input, output, eps_);
    return output;
}
