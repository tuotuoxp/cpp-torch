#pragma once
#include "../../include/nn/ReLU.h"
#include "Threshold.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class ReLU : public nn::ReLU<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                CHECK_AND_CAST(ReLU, Threshold, T)->unserialize(obj, mb);
            }
        };
    }
}
