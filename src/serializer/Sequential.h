#pragma once
#include "../../include/nn/Sequential.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class Sequential : public nn::Sequential<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(Sequential, Container, TTensor)->unserialize(obj, mb);
            }
        };
    }
}
