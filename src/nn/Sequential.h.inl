#pragma once
#include "../../include/nn/Sequential.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::Sequential<T>::forward(const cpptorch::Tensor<T> &input) const
{
    bool first = true;
    cpptorch::Tensor<T> out;
    for (auto &it : this->modules_)
    {
        out = it->forward(first ? input : out);
        first = false;
    }
    return out;
}
