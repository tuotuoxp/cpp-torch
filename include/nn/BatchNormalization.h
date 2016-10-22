#pragma once
#include "Layer.h"
#include "../torch/Tensor.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class BatchNormalization : public Layer<T,C>
        {
        public:
            BatchNormalization() : train_(false), ndim(2) {}

            virtual const std::string name() const override { return "nn.BatchNormalization"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            Tensor<T,C> weight_, bias_, running_mean_, running_var_;
            bool train_;
            double momentum_, eps_;

            // expected dimension of input
            int ndim;
        };
    }
}
