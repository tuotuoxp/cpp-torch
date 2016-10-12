#pragma once
#include "../../include/nn/Sequential.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class Sequential : public nn::Sequential<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb)
            {
                CHECK_AND_CAST(Sequential, Container, T)->unserialize(obj, mb);
            }
        };
    }
}
