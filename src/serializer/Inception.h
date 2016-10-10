#pragma once
#include "../../include/nn/Inception.h"
#include "Decorator.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class Inception : public nn::Inception<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(Inception, Decorator, TTensor)->unserialize(obj, mb);
            }
        };
    }
}
