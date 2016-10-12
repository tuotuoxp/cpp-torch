#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Threshold : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Threshold"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            T threshold_, val_;
            bool inplace_;
        };
    }
}
