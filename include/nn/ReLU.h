#pragma once
#include "Threshold.h"


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class ReLU : public Threshold<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.ReLU"; }
        };
    }
}
