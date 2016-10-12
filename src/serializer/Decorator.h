#pragma once
#include "../../include/nn/Decorator.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class Decorator : public nn::Decorator<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb)
            {
                CHECK_AND_CAST(Decorator, Container, T)->unserialize(obj, mb);
            }
        };
    }
}
