#pragma once
#include "../../include/nn/Sequential.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Sequential : public nn::Sequential<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                CHECK_AND_CAST(Sequential, Container, T)->unserialize(obj, mb);
            }
        };
    }
}
