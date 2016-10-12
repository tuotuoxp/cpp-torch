#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Decorator : public Container<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Decorator"; }
            virtual void print(std::ostream &o, int level) const override;
            virtual Tensor<T> forward(const Tensor<T> &input) const override;
        };
    }
}
