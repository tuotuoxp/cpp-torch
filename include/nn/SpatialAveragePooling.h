#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class SpatialAveragePooling : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialAveragePooling"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            int kW_, kH_, dW_, dH_, padW_, padH_;
            bool ceil_mode_, count_include_pad_, divide_;
        };
    }
}
