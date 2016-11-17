#pragma once
#include "Layer.h"
#include "../torch/Tensor.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class BatchNormalization : public Layer<T, F>
        {
        public:
            BatchNormalization() : train_(false), ndim(2) {}

            virtual const std::string name() const override { return "nn.BatchNormalization"; }
            API virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            Tensor<T, F> weight_, bias_, running_mean_, running_var_;
            bool train_;
            double momentum_, eps_;

            // expected dimension of input
            int ndim;
        };
    }
}
