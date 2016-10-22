#pragma once
#include "../../include/nn/Container.h"

#include <ostream>


template<typename T, bool C>
void cpptorch::nn::Container<T,C>::print(std::ostream &o, int level) const
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
