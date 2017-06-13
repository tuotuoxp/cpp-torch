#pragma once
#include "../../include/nn/Tanh.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Tanh : public nn::Tanh<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {

            }
        };
    }
}
