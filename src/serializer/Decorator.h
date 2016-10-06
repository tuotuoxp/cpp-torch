#pragma once
#include "../nn/Decorator.h"
#include "Container.h"


namespace serializer
{
    template<class TTensor>
    class Decorator : public nn::Decorator<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            CHECK_AND_CAST(Decorator, Container, TTensor)->unserialize(obj, mb);
        }
    };
}
