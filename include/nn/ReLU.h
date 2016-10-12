#pragma once
#include "Threshold.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class ReLU : public Threshold<T>
        {
        public:
            virtual const std::string name() const override { return "nn.ReLU"; }
        };
    }
}
