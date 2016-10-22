#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Square : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Square"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;
        };
    }
}
