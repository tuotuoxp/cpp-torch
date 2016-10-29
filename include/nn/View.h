#pragma once
#include "Layer.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class View : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.View"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            int num_elements_, num_input_dims_;
            std::vector<long> size_;
        };
    }
}
