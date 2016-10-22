#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Reshape : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Reshape"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            int nelement_;
            bool batch_mode_;
            std::vector<long> size_, batchsize_;
        };
    }
}
