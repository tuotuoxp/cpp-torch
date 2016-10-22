#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Threshold : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Threshold"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            T threshold_, val_;
            bool inplace_;
        };
    }
}
