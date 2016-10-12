#pragma once
#include "../../include/nn/Square.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::Square<T>::forward(const cpptorch::Tensor<T> &input) const
{
    cpptorch::Tensor<T> output(true);
    cpptorch::th::NN<T>::Square_updateOutput(nullptr, input, output);
    return output;
}
