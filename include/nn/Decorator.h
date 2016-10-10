#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Decorator : public Container<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Decorator"; }
            virtual void print(std::ostream &o, int level) const override;
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;
        };
    }
}
