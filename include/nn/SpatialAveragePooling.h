#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class SpatialAveragePooling : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialAveragePooling"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            int kW_, kH_, dW_, dH_, padW_, padH_;
            bool ceil_mode_, count_include_pad_, divide_;
        };
    }
}
