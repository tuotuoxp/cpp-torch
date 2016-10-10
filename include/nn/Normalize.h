#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Normalize : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Normalize"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            typename TTensor::Storage::Base p_, eps_;
        };
    }
}
