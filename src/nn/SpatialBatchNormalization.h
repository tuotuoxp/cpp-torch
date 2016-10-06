#pragma once
#include "BatchNormalization.h"


namespace nn
{
    template<class TTensor>
    class SpatialBatchNormalization : public BatchNormalization<TTensor>
    {
    public:
        SpatialBatchNormalization() { this->ndim = 4;  }

        virtual const std::string name() const override { return "nn.SpatialBatchNormalization"; }
    };
}
