#pragma once
#include "../../include/nn/SpatialLPPooling.h"
#include "Sequential.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class SpatialLPPooling : public nn::SpatialLPPooling<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(SpatialLPPooling, Sequential, TTensor)->unserialize(obj, mb);
            }
        };
    }
}
