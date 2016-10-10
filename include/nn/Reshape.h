#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Reshape : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Reshape"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            int nelement_;
            bool batch_mode_;
            std::vector<long> size_, batchsize_;
        };
    }
}
