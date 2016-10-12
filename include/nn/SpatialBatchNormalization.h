#pragma once
#include "BatchNormalization.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class SpatialBatchNormalization : public BatchNormalization<T>
        {
        public:
            SpatialBatchNormalization() { this->ndim = 4; }

            virtual const std::string name() const override { return "nn.SpatialBatchNormalization"; }
        };
    }
}
