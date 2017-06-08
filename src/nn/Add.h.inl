#pragma once
#include "../../include/nn/Add.h"
#include <iostream>

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Add<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> output(true);
    output.resizeAs(input);
    output.copy(input);
    if(scalar_)
    {
        cpptorch::Tensor<T, F> bias(true);
        bias.resizeAs(input);
        bias.fill(bias_[0]);
        output+=bias;
    }
    else
    {
        if(input.isSameSizeAs(bias_))
        {
            output+=bias_;
        }
        else
        {
            long batchSize=input.size(0);
            cpptorch::Tensor<T, F> bias = bias_.view({-1});
            cpptorch::Tensor<T, F> output_ref = output.view({batchSize, -1});
            if (_ones_.size(0) == batchSize)
            {
                output_ref.addr(1, _ones_, bias_);
            }
            else
            {
                cpptorch::Tensor<T, F> _ones(true);
                _ones.resize({batchSize});
                _ones.fill(1);
                output_ref.addr(1, _ones, bias);
            }

        }
    }
    return output;

}
