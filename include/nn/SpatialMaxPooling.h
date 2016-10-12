#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class SpatialMaxPooling : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialMaxPooling"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            int kW_, kH_, dW_, dH_, padW_, padH_;
            bool ceil_mode_;
        };
    }
}
