#pragma once
#include "Container.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Concat : public Container<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Concat"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            int dimension_;
        };
    }
}
