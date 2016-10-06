#pragma once
#include "../nn/Square.h"


namespace serializer
{
    template<class TTensor>
    class Square : public nn::Square<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb) {}
    };
}
