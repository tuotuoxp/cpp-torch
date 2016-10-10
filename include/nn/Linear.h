#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Linear : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Linear"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            Tensor<TTensor> weight_, bias_;
        };
    }
}
