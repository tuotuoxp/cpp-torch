#pragma once
#include "../../include/nn/SpatialLPPooling.h"
#include "Sequential.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class SpatialLPPooling : public nn::SpatialLPPooling<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb)
            {
                CHECK_AND_CAST(SpatialLPPooling, Sequential, T)->unserialize(obj, mb);
            }
        };
    }
}
