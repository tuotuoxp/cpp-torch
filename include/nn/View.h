#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class View : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.View"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            int num_elements_, num_input_dims_;
            std::vector<long> size_;
        };
    }
}
