#pragma once
#include "../../include/nn/Threshold.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Threshold<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    // validate parameters
    if (inplace_)
    {
        asserter(val_ <= threshold_) << "in-place processing requires value (" << val_ << ") not exceed threshold (" << threshold_ << ")";
    }

    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::NN<T, F>::Threshold_updateOutput(input, output, threshold_, val_, inplace_);
    return output;
}
