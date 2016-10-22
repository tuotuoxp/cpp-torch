#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class SpatialConvolution : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialConvolution"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            Tensor<T,C> weight_, bias_;
            int kW_, kH_, dW_, dH_, padW_, padH_;
        };
    }
}
