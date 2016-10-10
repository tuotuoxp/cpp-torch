#pragma once
#include "../../include/nn/SpatialBatchNormalization.h"
#include "BatchNormalization.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class SpatialBatchNormalization : public nn::SpatialBatchNormalization<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(SpatialBatchNormalization, BatchNormalization, TTensor)->unserialize(obj, mb);
            }
        };
    }
}
