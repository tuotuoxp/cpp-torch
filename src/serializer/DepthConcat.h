#pragma once
#include "../../include/nn/DepthConcat.h"
#include "Concat.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class DepthConcat : public nn::DepthConcat<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                CHECK_AND_CAST(DepthConcat, Concat, T)->unserialize(obj, mb);
            }
        };
    }
}
