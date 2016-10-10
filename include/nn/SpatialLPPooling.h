#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class SpatialLPPooling : public Sequential<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialLPPooling"; }
        };
    }
}
