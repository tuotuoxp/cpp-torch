#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Sequential : public Container<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Sequential"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;
        };
    }
}
