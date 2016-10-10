#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class SpatialReflectionPadding : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialReflectionPadding"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            int pad_l_, pad_r_, pad_t_, pad_b_;
        };
    }
}
