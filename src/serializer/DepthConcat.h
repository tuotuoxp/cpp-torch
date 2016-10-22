#pragma once
#include "../../include/nn/DepthConcat.h"
#include "Concat.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class DepthConcat : public nn::DepthConcat<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                CHECK_AND_CAST(DepthConcat, Concat, T)->unserialize(obj, mb);
            }
        };
    }
}
