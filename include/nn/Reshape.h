#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Reshape : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Reshape"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            int nelement_;
            bool batch_mode_;
            std::vector<long> size_, batchsize_;
        };
    }
}
