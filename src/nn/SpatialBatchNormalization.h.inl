#pragma once
#include "BatchNormalization.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class SpatialBatchNormalization : public BatchNormalization<T, F>
        {
        public:
            SpatialBatchNormalization() { this->ndim = 4; }

            virtual const std::string name() const override { return "nn.SpatialBatchNormalization"; }
        };
    }
}
