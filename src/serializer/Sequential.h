#pragma once
#include "../../include/nn/Sequential.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class Sequential : public nn::Sequential<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                CHECK_AND_CAST(Sequential, Container, T)->unserialize(obj, mb);
            }
        };
    }
}
