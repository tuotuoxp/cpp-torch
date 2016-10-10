#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Sequential : public Container<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Sequential"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;
        };
    }
}
