#pragma once
#include "../../include/nn/Decorator.h"


template<typename T>
void cpptorch::nn::Decorator<T>::print(std::ostream &o, int level) const
{
    o << name() << " @ ";
    this->modules_[0]->print(o, level);
}

template<typename T>
cpptorch::Tensor<T> cpptorch::nn::Decorator<T>::forward(const cpptorch::Tensor<T> &input) const
{
    return this->modules_[0]->forward(input);
}
