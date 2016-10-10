#pragma once
#include "../../include/nn/Decorator.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class Decorator : public nn::Decorator<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(Decorator, Container, TTensor)->unserialize(obj, mb);
            }
        };
    }
}
