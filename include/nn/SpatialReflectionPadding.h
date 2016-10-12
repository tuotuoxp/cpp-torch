#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class SpatialReflectionPadding : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialReflectionPadding"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            int pad_l_, pad_r_, pad_t_, pad_b_;
        };
    }
}
