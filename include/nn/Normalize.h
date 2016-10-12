#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Normalize : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Normalize"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            T p_, eps_;
        };
    }
}
