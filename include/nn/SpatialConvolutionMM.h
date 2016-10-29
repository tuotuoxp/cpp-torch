#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class SpatialConvolutionMM : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialConvolutionMM"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            Tensor<T, F> weight_, bias_;
            int kW_, kH_, dW_, dH_, padW_, padH_;
        };
    }
}
