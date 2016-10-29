#pragma once
#include "../../include/nn/Sequential.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Sequential<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    bool first = true;
    cpptorch::Tensor<T, F> out;
    for (auto &it : this->modules_)
    {
        out = it->forward(first ? input : out);
        first = false;
    }
    return out;
}
