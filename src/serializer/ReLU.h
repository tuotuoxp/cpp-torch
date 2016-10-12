#pragma once
#include "../../include/nn/ReLU.h"
#include "Threshold.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class ReLU : public nn::ReLU<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb)
            {
                CHECK_AND_CAST(ReLU, Threshold, T)->unserialize(obj, mb);
            }
        };
    }
}
