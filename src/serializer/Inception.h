#pragma once
#include "../../include/nn/Inception.h"
#include "Decorator.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class Inception : public nn::Inception<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                CHECK_AND_CAST(Inception, Decorator, T)->unserialize(obj, mb);
            }
        };
    }
}
