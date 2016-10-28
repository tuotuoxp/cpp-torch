#pragma once
#include "../../include/nn/Threshold.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::Threshold<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    // validate parameters
    if (inplace_)
    {
        asserter(val_ <= threshold_) << "in-place processing requires value (" << val_ << ") not exceed threshold (" << threshold_ << ")";
    }

    cpptorch::Tensor<T,C> output(true);
    cpptorch::th::NN<T,C>::Threshold_updateOutput(input, output, threshold_, val_, inplace_);
    return output;
}
