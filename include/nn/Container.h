#pragma once
#include "Layer.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<class TTensor>
        class Container : public Layer<TTensor>
        {
        public:
            virtual const std::string name() const override { return "nn.Container"; }
            virtual void print(std::ostream &o, int level) const override;

        protected:
            std::vector<std::shared_ptr<Layer<TTensor>>> modules_;
        };
    }
}
