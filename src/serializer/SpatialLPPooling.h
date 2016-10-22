#pragma once
#include "../../include/nn/SpatialLPPooling.h"
#include "Sequential.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class SpatialLPPooling : public nn::SpatialLPPooling<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                CHECK_AND_CAST(SpatialLPPooling, Sequential, T)->unserialize(obj, mb);
            }
        };
    }
}
