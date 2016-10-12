#pragma once
#include "Layer.h"
#include "../torch/Tensor.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class BatchNormalization : public Layer<T>
        {
        public:
            BatchNormalization() : train_(false), ndim(2) {}

            virtual const std::string name() const override { return "nn.BatchNormalization"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            Tensor<T> weight_, bias_, running_mean_, running_var_;
            bool train_;
            double momentum_, eps_;

            // expected dimension of input
            int ndim;
        };
    }
}
