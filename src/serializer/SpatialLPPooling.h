#pragma once
#include "../nn/SpatialLPPooling.h"
#include "Sequential.h"


namespace serializer
{
    template<class TTensor>
    class SpatialLPPooling : public nn::SpatialLPPooling<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            CHECK_AND_CAST(SpatialLPPooling, Sequential, TTensor)->unserialize(obj, mb);
        }
    };
}
