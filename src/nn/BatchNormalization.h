#pragma once
#include "Layer.h"
#include "Tensor.h"


namespace nn
{
    template<class TTensor>
    class BatchNormalization : public Layer<TTensor>
    {
    public:
        BatchNormalization() : train_(false), ndim(2) {}

        virtual const std::string name() const override { return "nn.BatchNormalization"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

    protected:
        Tensor<TTensor> weight_, bias_, running_mean_, running_var_;
        bool train_;
        double momentum_, eps_;

        // expected dimension of input
        int ndim;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::BatchNormalization<TTensor>::forward(const nn::Tensor<TTensor> &input) const
{
    int idim = input.dim();

    // check input dim
    asserter(idim == ndim || (idim == ndim - 1 && train_ == false))
        << "only mini-batch supported (" << ndim << "D tensor), got " << idim << "D tensor instead";
    int feat_dim = (idim == ndim - 1) ? 0 : 1;
    asserter(input.size(feat_dim) == running_mean_.nElement())
        << "got " << input.size(feat_dim) << "-feature tensor, expected " << running_mean_.nElement();

    // make input contiguous
    nn::Tensor<TTensor> input_new;
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

    nn::Tensor<TTensor> save_mean(true), save_std(true);
    save_mean.resizeAs(running_mean_);
    save_std.resizeAs(running_var_);

    nn::Tensor<TTensor> output(true);
    output.resizeAs(input_new.valid() ? input_new : input);
    THWrapper::NN<TTensor>::BatchNormalization_updateOutput(nullptr, input_new.valid() ? input_new : input,
        output, weight_, bias_, running_mean_, running_var_, save_mean, save_std, train_, momentum_, eps_);
    return output;
}
