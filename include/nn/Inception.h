#pragma once
#include "Decorator.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Inception : public Decorator<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Inception"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;
        };
    }
}
