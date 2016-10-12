#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class DepthConcat : public Concat<T>
        {
        public:
            virtual const std::string name() const override { return "nn.DepthConcat"; }
            virtual Tensor<T> forward(const Tensor<T> &input) const override;

        protected:
            Tensor<T> windowNarrow(Tensor<T> &output, Tensor<T> &currentOutput, std::vector<long> &outputSize, int offset) const;
        };
    }
}
