#pragma once
#include "../../include/nn/Square.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class Square : public nn::Square<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb) {}
        };
    }
}
