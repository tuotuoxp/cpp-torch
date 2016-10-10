#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class MulConstant : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.MulConstant"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            bool inplace_;
            typename TTensor::Storage::Base constant_scalar_;
        };
    }
}
