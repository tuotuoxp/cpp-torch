#pragma once
#include "../../include/nn/Sequential.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::Sequential<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    bool first = true;
    cpptorch::Tensor<T,C> out;
    for (auto &it : this->modules_)
    {
        out = it->forward(first ? input : out);
        first = false;
    }
    return out;
}
