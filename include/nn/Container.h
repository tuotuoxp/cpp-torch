#pragma once
#include "Layer.h"

#include <vector>


namespace cpptorch
{
    namespace nn
    {
        template<typename T, GPUFlag F>
        class Container : public Layer<T, F>
        {
        public:
            virtual const std::string name() const override { return "nn.Container"; }
            virtual void print(std::ostream &o, int level) const override;
            virtual std::shared_ptr<Layer<T, F>> get(int level) { return modules_[level]; };
        protected:
            std::vector<std::shared_ptr<Layer<T, F>>> modules_;
        };
    }
}
