#pragma once
#include "Layer.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<typename T>
        class Container : public Layer<T>
        {
        public:
            virtual const std::string name() const override { return "nn.Container"; }
            virtual void print(std::ostream &o, int level) const override;

        protected:
            std::vector<std::shared_ptr<Layer<T>>> modules_;
        };
    }
}
