#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class SpatialConvolution : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.SpatialConvolution"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

    protected:
        Tensor<TTensor> weight_, bias_;
        int kW_, kH_, dW_, dH_, padW_, padH_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::SpatialConvolution<TTensor>::forward(const nn::Tensor<TTensor> &input) const
{
    nn::Tensor<TTensor> finput(true), fgradinput(true);
    nn::Tensor<TTensor> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }

    nn::Tensor<TTensor> out(true);
    THWrapper::NN<TTensor>::SpatialConvolutionMM_updateOutput(nullptr, input_new.valid() ? input_new : input, 
        out, weight_, bias_, finput, fgradinput, kW_, kH_, dW_, dH_, padW_, padH_);
    return out;
}

