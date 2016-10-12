#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Linear : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Linear"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            Tensor<T> weight_, bias_;
        };
    }
}
