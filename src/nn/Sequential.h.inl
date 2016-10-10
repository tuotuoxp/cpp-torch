#pragma once
#include "../../include/nn/Sequential.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Sequential<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    bool first = true;
    cpptorch::Tensor<TTensor> out;
    for (auto &it : this->modules_)
    {
        out = it->forward(first ? input : out);
        first = false;
    }
    return out;
}
