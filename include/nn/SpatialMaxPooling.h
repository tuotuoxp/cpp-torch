#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class SpatialMaxPooling : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialMaxPooling"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            int kW_, kH_, dW_, dH_, padW_, padH_;
            bool ceil_mode_;
        };
    }
}
