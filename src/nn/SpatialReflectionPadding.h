#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class SpatialReflectionPadding : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.SpatialReflectionPadding"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

    protected:
        int pad_l_, pad_r_, pad_t_, pad_b_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::SpatialReflectionPadding<TTensor>::forward(const nn::Tensor<TTensor> &input) const
{
    int idim = input.dim();
    assert((idim == 3 || idim == 4) && "input must be 3 or 4-dimensional");
    nn::Tensor<TTensor> output(true);
    THWrapper::NN<TTensor>::SpatialReflectionPadding_updateOutput(nullptr, input, output, pad_l_, pad_r_, pad_t_, pad_b_);
    return output;
}
