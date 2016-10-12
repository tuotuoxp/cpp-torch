#pragma once
#include "../../include/nn/SpatialBatchNormalization.h"
#include "BatchNormalization.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class SpatialBatchNormalization : public nn::SpatialBatchNormalization<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb)
            {
                CHECK_AND_CAST(SpatialBatchNormalization, BatchNormalization, T)->unserialize(obj, mb);
            }
        };
    }
}
