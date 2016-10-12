#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class SpatialCrossMapLRN : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialCrossMapLRN"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            int size_;
            T alpha_, beta_, k_;
        };
    }
}
