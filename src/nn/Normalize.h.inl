#pragma once
#include "../../include/nn/Normalize.h"

#include <cmath>


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Normalize<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    assert(input.dim() <= 2 && "only 1d layer supported");
    std::vector<long> input_size = input.size();
    cpptorch::Tensor<TTensor> input_new;
    if (input.dim() == 1)
    {
        input_new = input.view({ 1, -1 });
    }
    else
    {
        input_new = input;
    }

    cpptorch::Tensor<TTensor> norm;
    if (std::isinf((double)p_))
    {
        // specialization for the infinity norm
        norm = cpptorch::abs(input_new).max(1) + eps_;
    }
    else
    {
        cpptorch::Tensor<TTensor> buffer;
        if ((int)p_ % 2 != 0)
        {
            buffer = cpptorch::abs(input_new) ^ p_;
        }
        else
        {
            buffer = input_new ^ p_;
        }
        norm = (buffer.sum(1) + eps_) ^ (1 / p_);
    }

    cpptorch::Tensor<TTensor> output;
    output = (input / norm.view({ -1, 1 }).expand(input_new.size())).view(input_size);
    return output;
}
