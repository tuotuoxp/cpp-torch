#pragma once
#include "Threshold.h"


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class ReLU : public Threshold<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.ReLU"; }
        };
    }
}
