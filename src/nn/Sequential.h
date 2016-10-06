#pragma once
#include "Container.h"


namespace nn
{
    template<class TTensor>
    class Sequential : public Container<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Sequential"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Sequential<TTensor>::forward(const Tensor<TTensor> &input)
{
    bool first = true;
    Tensor<TTensor> out;
    for (auto &it : this->modules_)
    {
        out = it->forward(first ? input : out);
        first = false;
    }
    return out;
}
