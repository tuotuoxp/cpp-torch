#pragma once
#include "../../include/nn/Decorator.h"


template<typename T, GPUFlag F>
void cpptorch::nn::Decorator<T, F>::print(std::ostream &o, int level) const
{
    o << name() << " @ ";
    this->modules_[0]->print(o, level);
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Decorator<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    return this->modules_[0]->forward(input);
}
