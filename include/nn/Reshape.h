#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Reshape : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Reshape"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            int nelement_;
            bool batch_mode_;
            std::vector<long> size_, batchsize_;
        };
    }
}
