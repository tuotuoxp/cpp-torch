#pragma once
#include "../nn/SpatialBatchNormalization.h"
#include "BatchNormalization.h"


namespace serializer
{
    template<class TTensor>
    class SpatialBatchNormalization : public nn::SpatialBatchNormalization<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            CHECK_AND_CAST(SpatialBatchNormalization, BatchNormalization, TTensor)->unserialize(obj, mb);
        }
    };
}
