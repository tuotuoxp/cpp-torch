#pragma once
#include "../nn/DepthConcat.h"
#include "Concat.h"


namespace serializer
{
    template<class TTensor>
    class DepthConcat : public nn::DepthConcat<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            CHECK_AND_CAST(DepthConcat, Concat, TTensor)->unserialize(obj, mb);
        }
    };
}
