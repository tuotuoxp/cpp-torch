#pragma once
#include "../nn/Sequential.h"
#include "Container.h"


namespace serializer
{
    template<class TTensor>
    class Sequential : public nn::Sequential<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            CHECK_AND_CAST(Sequential, Container, TTensor)->unserialize(obj, mb);
        }
    };
}
