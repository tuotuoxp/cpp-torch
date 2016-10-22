#pragma once
#include "../../include/nn/Square.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::Square<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    cpptorch::Tensor<T,C> output(true);
    cpptorch::th::NN<T,C>::Square_updateOutput(nullptr, input, output);
    return output;
}
