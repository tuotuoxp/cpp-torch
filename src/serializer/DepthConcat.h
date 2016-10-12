#pragma once
#include "../../include/nn/DepthConcat.h"
#include "Concat.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class DepthConcat : public nn::DepthConcat<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb)
            {
                CHECK_AND_CAST(DepthConcat, Concat, T)->unserialize(obj, mb);
            }
        };
    }
}
