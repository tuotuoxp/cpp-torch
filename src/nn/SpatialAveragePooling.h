#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class SpatialAveragePooling : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.SpatialAveragePooling"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

    protected:
        int kW_, kH_, dW_, dH_, padW_, padH_;
        bool ceil_mode_, count_include_pad_, divide_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::SpatialAveragePooling<TTensor>::forward(const nn::Tensor<TTensor> &input) const
{
    nn::Tensor<TTensor> output(true);
    THWrapper::NN<TTensor>::SpatialAveragePooling_updateOutput(nullptr, input, output,
        kW_, kH_, dW_, dH_, padW_, padH_, ceil_mode_, count_include_pad_);

    // for backward compatibility with saved models which are not supposed to have "divide" field
    if (!divide_)
    {
        output *= (typename TTensor::Storage::StorageBase)(kW_ * kH_);
    }
    return output;
}
