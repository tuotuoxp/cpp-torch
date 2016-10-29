#pragma once
#include "../../include/nn/Decorator.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Decorator : public nn::Decorator<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                CHECK_AND_CAST(Decorator, Container, T)->unserialize(obj, mb);
            }
        };
    }
}
