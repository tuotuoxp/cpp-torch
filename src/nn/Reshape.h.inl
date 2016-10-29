#pragma once
#include "../../include/nn/Reshape.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Reshape<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }
    else
    {
        input_new = input;
    }
    cpptorch::Tensor<T, F> output;
    if (!batch_mode_ && input_new.nElement() == nelement_ && input_new.size(0) != 1)
    {
        output = input_new.view(size_);
    }
    else
    {
        std::vector<long> batchsize = batchsize_;
        batchsize[0] = input_new.size(0);
        output = input_new.view(batchsize);
    }
    return output;
}
