#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class SoftMax : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.SoftMax"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:

        };
    }
}
