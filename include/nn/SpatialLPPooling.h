#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class SpatialLPPooling : public Sequential<T>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialLPPooling"; }
        };
    }
}
