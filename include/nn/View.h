#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class View : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.View"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            int num_elements_, num_input_dims_;
            std::vector<long> size_;
        };
    }
}
