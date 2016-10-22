#pragma once
#include "../../include/nn/Sqrt.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::Sqrt<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    cpptorch::Tensor<T,C> output(true);
    cpptorch::th::NN<T,C>::Sqrt_updateOutput(nullptr, input, output, eps_);
    return output;
}
