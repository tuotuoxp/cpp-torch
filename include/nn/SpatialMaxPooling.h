#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class SpatialMaxPooling : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialMaxPooling"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            int kW_, kH_, dW_, dH_, padW_, padH_;
            bool ceil_mode_;
        };
    }
}
