#pragma once
#include "Container.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Concat : public Container<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Concat"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            int dimension_;
        };
    }
}
