#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Linear : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Linear"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            Tensor<T, F> weight_, bias_;
        };
    }
}
