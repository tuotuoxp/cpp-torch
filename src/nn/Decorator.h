#pragma once
#include "Container.h"


namespace nn
{
    template<class TTensor>
    class Decorator : public Container<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Decorator"; }
        virtual void print(std::ostream &o, int level) const override;
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;
    };
}


template<class TTensor>
void nn::Decorator<TTensor>::print(std::ostream &o, int level) const
{
    o << name() << " @ ";
    this->modules_[0]->print(o, level);
}

template<class TTensor>
nn::Tensor<TTensor> nn::Decorator<TTensor>::forward(const nn::Tensor<TTensor> &input) const
{
    return this->modules_[0]->forward(input);
}
