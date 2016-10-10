#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class View : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.View"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            int num_elements_, num_input_dims_;
            std::vector<long> size_;
        };
    }
}
