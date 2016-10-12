#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Square : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Square"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;
        };
    }
}
