#pragma once
#include "../../include/nn/BatchNormalization.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::BatchNormalization<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    int idim = input.dim();

    // check input dim
    asserter(idim == ndim || (idim == ndim - 1 && train_ == false))
        << "only mini-batch supported (" << ndim << "D tensor), got " << idim << "D tensor instead";
    int feat_dim = (idim == ndim - 1) ? 0 : 1;
    asserter(input.size(feat_dim) == running_mean_.nElement())
        << "got " << input.size(feat_dim) << "-feature tensor, expected " << running_mean_.nElement();

    // make input contiguous
    cpptorch::Tensor<TTensor> input_new;
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

    cpptorch::Tensor<TTensor> save_mean(true), save_std(true);
    save_mean.resizeAs(running_mean_);
    save_std.resizeAs(running_var_);

    cpptorch::Tensor<TTensor> output(true);
    output.resizeAs(input_new.valid() ? input_new : input);
    cpptorch::th::NN<TTensor>::BatchNormalization_updateOutput(nullptr, input_new.valid() ? input_new : input,
        output, weight_, bias_, running_mean_, running_var_, save_mean, save_std, train_, momentum_, eps_);
    return output;
}
