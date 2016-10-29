#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Sequential : public Container<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Sequential"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;
        };
    }
}
