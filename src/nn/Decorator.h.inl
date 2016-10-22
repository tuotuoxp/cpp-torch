#pragma once
#include "../../include/nn/Decorator.h"


template<typename T, bool C>
void cpptorch::nn::Decorator<T,C>::print(std::ostream &o, int level) const
{
    o << name() << " @ ";
    this->modules_[0]->print(o, level);
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::Decorator<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    return this->modules_[0]->forward(input);
}
