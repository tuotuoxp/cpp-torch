#pragma once
#include "../../include/nn/Threshold.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Threshold<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    // validate parameters
    if (inplace_)
    {
        asserter(val_ <= threshold_) << "in-place processing requires value (" << val_ << ") not exceed threshold (" << threshold_ << ")";
    }

    cpptorch::Tensor<TTensor> output(true);
    cpptorch::th::NN<TTensor>::Threshold_updateOutput(nullptr, input, output, threshold_, val_, inplace_);
    return output;
}
