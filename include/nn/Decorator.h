#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Decorator : public Container<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Decorator"; }
            virtual void print(std::ostream &o, int level) const override;
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;
        };
    }
}
