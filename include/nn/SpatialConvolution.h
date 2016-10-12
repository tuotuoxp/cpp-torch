#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class SpatialConvolution : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialConvolution"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            Tensor<T> weight_, bias_;
            int kW_, kH_, dW_, dH_, padW_, padH_;
        };
    }
}
