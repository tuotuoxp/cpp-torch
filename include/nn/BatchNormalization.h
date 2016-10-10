#pragma once
#include "Layer.h"
#include "../torch/Tensor.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class BatchNormalization : public Layer<TTensor>
        {
        public:
            BatchNormalization() : train_(false), ndim(2) {}

            virtual const std::string name() const override { return "nn.BatchNormalization"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            Tensor<TTensor> weight_, bias_, running_mean_, running_var_;
            bool train_;
            double momentum_, eps_;

            // expected dimension of input
            int ndim;
        };
    }
}
