#pragma once
#include "../../include/nn/Reshape.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::Reshape<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    cpptorch::Tensor<T,C> input_new;
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
    cpptorch::Tensor<T,C> output;
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
