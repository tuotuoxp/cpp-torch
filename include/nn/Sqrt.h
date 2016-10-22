#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Sqrt : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Sqrt"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            T eps_;
        };
    }
}
