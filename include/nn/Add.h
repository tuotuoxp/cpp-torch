#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Add : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Add"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            bool scalar_;
            Tensor<T, F> bias_, gradBias_, _ones_;
        };
    }
}
