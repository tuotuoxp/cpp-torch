#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class SpatialMaxPooling : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.SpatialMaxPooling"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;

    protected:
        int kW_, kH_, dW_, dH_, padW_, padH_;
        bool ceil_mode_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::SpatialMaxPooling<TTensor>::forward(const nn::Tensor<TTensor> &input)
{
    nn::Tensor<TTensor> output(true), indices(true);
    THWrapper::NN<TTensor>::SpatialMaxPooling_updateOutput(nullptr, input, output, indices, kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_);
    return output;
}
