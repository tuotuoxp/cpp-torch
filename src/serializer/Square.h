#pragma once
#include "../../include/nn/Square.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Square : public nn::Square<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb) {}
        };
    }
}
