#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class DepthConcat : public Concat<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.DepthConcat"; }
            virtual Tensor<T, F> forward(const Tensor<T, F> &input) const override;

        protected:
            Tensor<T, F> windowNarrow(Tensor<T, F> &output, Tensor<T, F> &currentOutput, std::vector<long> &outputSize, int offset) const;
        };
    }
}
