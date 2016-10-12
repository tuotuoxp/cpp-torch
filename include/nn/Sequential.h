#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Sequential : public Container<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Sequential"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;
        };
    }
}
