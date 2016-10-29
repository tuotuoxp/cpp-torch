#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class SpatialLPPooling : public Sequential<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialLPPooling"; }
        };
    }
}
