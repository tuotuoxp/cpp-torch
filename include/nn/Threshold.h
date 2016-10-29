#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Threshold : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Threshold"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            T threshold_, val_;
            bool inplace_;
        };
    }
}
