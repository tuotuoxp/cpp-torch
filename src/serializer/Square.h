#pragma once
#include "../../include/nn/Square.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class Square : public nn::Square<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb) {}
        };
    }
}
