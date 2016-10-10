#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Square : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Square"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;
        };
    }
}
