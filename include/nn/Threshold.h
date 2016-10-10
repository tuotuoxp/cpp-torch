#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Threshold : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Threshold"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            typename TTensor::Storage::Base threshold_, val_;
            bool inplace_;
        };
    }
}
