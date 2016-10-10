#pragma once
#include "../../include/nn/Decorator.h"


template<class TTensor>
void cpptorch::nn::Decorator<TTensor>::print(std::ostream &o, int level) const
{
    o << name() << " @ ";
    this->modules_[0]->print(o, level);
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Decorator<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    return this->modules_[0]->forward(input);
}
