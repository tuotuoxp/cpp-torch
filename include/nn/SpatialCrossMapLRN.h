#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class SpatialCrossMapLRN : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialCrossMapLRN"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            int size_;
            T alpha_, beta_, k_;
        };
    }
}
