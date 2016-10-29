#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Decorator : public Container<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Decorator"; }
            virtual void print(std::ostream &o, int level) const override;
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;
        };
    }
}
