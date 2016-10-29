#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class SpatialCrossMapLRN : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialCrossMapLRN"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            int size_;
            T alpha_, beta_, k_;
        };
    }
}
