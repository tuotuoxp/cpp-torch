#pragma once
#include "../../include/nn/Threshold.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::Threshold<T>::forward(const cpptorch::Tensor<T> &input) const
{
    // validate parameters
    if (inplace_)
    {
        asserter(val_ <= threshold_) << "in-place processing requires value (" << val_ << ") not exceed threshold (" << threshold_ << ")";
    }

    cpptorch::Tensor<T> output(true);
    cpptorch::th::NN<T>::Threshold_updateOutput(nullptr, input, output, threshold_, val_, inplace_);
    return output;
}
