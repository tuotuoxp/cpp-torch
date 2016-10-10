#pragma once
#include "Container.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Concat : public Container<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Concat"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            int dimension_;
        };
    }
}
