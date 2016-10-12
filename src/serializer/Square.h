#pragma once
#include "../../include/nn/Square.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class Square : public nn::Square<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb) {}
        };
    }
}
