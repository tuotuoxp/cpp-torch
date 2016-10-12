#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Sqrt : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Sqrt"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            T eps_;
        };
    }
}
