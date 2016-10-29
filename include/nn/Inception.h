#pragma once
#include "Decorator.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Inception : public Decorator<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Inception"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;
        };
    }
}
