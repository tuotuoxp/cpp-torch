#pragma once
#include "../../include/nn/Inception.h"
#include "Decorator.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Inception : public nn::Inception<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                CHECK_AND_CAST(Inception, Decorator, T)->unserialize(obj, mb);
            }
        };
    }
}
