#pragma once
#include "../../include/nn/DepthConcat.h"
#include "Concat.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class DepthConcat : public nn::DepthConcat<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(DepthConcat, Concat, TTensor)->unserialize(obj, mb);
            }
        };
    }
}
