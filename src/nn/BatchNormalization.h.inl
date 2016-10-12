#pragma once
#include "../../include/nn/BatchNormalization.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::BatchNormalization<T>::forward(const cpptorch::Tensor<T> &input) const
{
    int idim = input.dim();

    // check input dim
    asserter(idim == ndim || (idim == ndim - 1 && train_ == false))
        << "only mini-batch supported (" << ndim << "D tensor), got " << idim << "D tensor instead";
    int feat_dim = (idim == ndim - 1) ? 0 : 1;
    asserter(input.size(feat_dim) == running_mean_.nElement())
        << "got " << input.size(feat_dim) << "-feature tensor, expected " << running_mean_.nElement();

    // make input contiguous
    cpptorch::Tensor<T> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }

    // make batch
    if (train_ == false && idim == ndim - 1)
    {
        input_new = addSingletonDimension(input_new.valid() ? input_new : input, 1);
    }

    cpptorch::Tensor<T> save_mean(true), save_std(true);
    save_mean.resizeAs(running_mean_);
    save_std.resizeAs(running_var_);

    cpptorch::Tensor<T> output(true);
    output.resizeAs(input_new.valid() ? input_new : input);
    cpptorch::th::NN<T>::BatchNormalization_updateOutput(nullptr, input_new.valid() ? input_new : input,
        output, weight_, bias_, running_mean_, running_var_, save_mean, save_std, train_, momentum_, eps_);
    return output;
}
