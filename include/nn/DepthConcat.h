#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class DepthConcat : public Concat<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.DepthConcat"; }
            virtual Tensor<T,C> forward(const Tensor<T,C> &input) const override;

        protected:
            Tensor<T,C> windowNarrow(Tensor<T,C> &output, Tensor<T,C> &currentOutput, std::vector<long> &outputSize, int offset) const;
        };
    }
}
