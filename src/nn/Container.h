#pragma once
#include "Layer.h"

#include <vector>
#include <ostream>


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


template<class TTensor>
void nn::Container<TTensor>::print(std::ostream &o, int level) const
{
    o << name() << " {" << std::endl;
    int counter = 1;
    for (auto &it_mod : modules_)
    {
        o << std::string(3 * (level + 1), ' ') << "(" << counter++ << ") ";
        it_mod->print(o, level + 1);
        o << std::endl;
    }
    o << std::string(3 * level, ' ') << "}";
}
