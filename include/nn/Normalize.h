#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Normalize : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Normalize"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            T p_, eps_;
        };
    }
}
