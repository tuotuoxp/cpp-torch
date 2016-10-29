#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class SpatialReflectionPadding : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialReflectionPadding"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            int pad_l_, pad_r_, pad_t_, pad_b_;
        };
    }
}
