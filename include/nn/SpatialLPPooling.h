#pragma once
#include "Sequential.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class SpatialLPPooling : public Sequential<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.SpatialLPPooling"; }
        };
    }
}
