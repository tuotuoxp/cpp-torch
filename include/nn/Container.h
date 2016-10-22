#pragma once
#include "Layer.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<typename T, bool C>
        class Container : public Layer<T,C>
        {
        public:
            virtual const std::string name() const override { return "nn.Container"; }
            virtual void print(std::ostream &o, int level) const override;

        protected:
            std::vector<std::shared_ptr<Layer<T,C>>> modules_;
        };
    }
}
