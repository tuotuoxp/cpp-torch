#pragma once
#include "Container.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Concat : public Container<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Concat"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            int dimension_;
        };
    }
}
