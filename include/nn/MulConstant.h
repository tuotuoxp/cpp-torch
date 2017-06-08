#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class MulConstant : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.MulConstant"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;
            virtual void setConstant(const T val);
        protected:
            bool inplace_;
            T constant_scalar_;
        };
    }
}
