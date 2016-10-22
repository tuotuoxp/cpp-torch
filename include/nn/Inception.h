#pragma once
#include "Decorator.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Inception : public Decorator<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Inception"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;
        };
    }
}
