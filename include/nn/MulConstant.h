#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class MulConstant : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.MulConstant"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            bool inplace_;
            T constant_scalar_;
        };
    }
}
