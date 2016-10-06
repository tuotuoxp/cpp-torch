#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class Square : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Square"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Square<TTensor>::forward(const nn::Tensor<TTensor> &input)
{
    nn::Tensor<TTensor> output(true);
    THWrapper::NN<TTensor>::Square_updateOutput(nullptr, input, output);
    return output;
}
