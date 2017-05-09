#pragma once
#include "../../include/nn/SoftMax.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class SoftMax : public nn::SoftMax<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {

            }
        };
    }
}
