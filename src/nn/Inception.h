#pragma once
#include "Decorator.h"


namespace nn
{
    template<class TTensor>
    class Inception : public Decorator<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Inception"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Inception<TTensor>::forward(const nn::Tensor<TTensor> &input)
{
    if (input.dim() == 3)
    {
        return fromBatch(Decorator<TTensor>::forward(toBatch(input)));
    }
    else
    {
        return Decorator<TTensor>::forward(input);
    }
}
