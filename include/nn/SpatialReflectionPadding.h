#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class SpatialReflectionPadding : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialReflectionPadding"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            int pad_l_, pad_r_, pad_t_, pad_b_;
        };
    }
}
