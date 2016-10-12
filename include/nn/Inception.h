#pragma once
#include "Decorator.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Inception : public Decorator<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Inception"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;
        };
    }
}
