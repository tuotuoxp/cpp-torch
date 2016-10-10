#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class SpatialCrossMapLRN : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialCrossMapLRN"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            int size_;
            typename TTensor::Storage::Base alpha_, beta_, k_;
        };
    }
}
