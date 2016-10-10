#pragma once
#include "Container.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class DepthConcat : public Concat<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.DepthConcat"; }
            virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

        protected:
            Tensor<TTensor> windowNarrow(Tensor<TTensor> &output, Tensor<TTensor> &currentOutput, std::vector<long> &outputSize, int offset) const;
        };
    }
}
