#pragma once
#include "../../include/nn/SpatialLPPooling.h"
#include "Sequential.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class SpatialLPPooling : public nn::SpatialLPPooling<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                CHECK_AND_CAST(SpatialLPPooling, Sequential, T)->unserialize(obj, mb);
            }
        };
    }
}
