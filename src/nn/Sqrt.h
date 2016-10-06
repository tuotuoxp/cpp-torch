#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class Sqrt : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Sqrt"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;

    protected:
        typename TTensor::Storage::StorageBase eps_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Sqrt<TTensor>::forward(const nn::Tensor<TTensor> &input)
{
    nn::Tensor<TTensor> output(true);
    THWrapper::NN<TTensor>::Sqrt_updateOutput(nullptr, input, output, eps_);
    return output;
}
