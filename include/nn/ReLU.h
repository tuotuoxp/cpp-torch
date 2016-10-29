#pragma once
#include "Threshold.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class ReLU : public Threshold<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.ReLU"; }
        };
    }
}
