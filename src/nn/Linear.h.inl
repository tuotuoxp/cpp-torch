#pragma once
#include "../../include/nn/Linear.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Linear<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    cpptorch::Tensor<T, F> output(true);
    int idim = input.dim();
    if (idim == 1)
    {
        output.resize({ weight_.size(0) });
        if (bias_.valid())
        {
            output.copy(bias_);
        }
        else
        {
            output.fill(0);
        }
        output.addmv(1, weight_, input);
    }
    else if (idim == 2)
    {
        long nframe = input.size(0);
        long nElement = output.nElement();
        output.resize({ nframe, weight_.size(0) });
        if (output.nElement() != nElement)
        {
            output.fill(0);
        }

        cpptorch::Tensor<T, F> addBuffer(true);
        addBuffer.resize({ nframe });
        addBuffer.fill(1);
        output.addmm(0, output, 1, input, weight_.t());
        if (bias_.valid())
        {
            output.addr(1, addBuffer, bias_);
        }
    }
    else
    {
        assert(0 && "input must be vector or matrix");
    }
    return output;
}
