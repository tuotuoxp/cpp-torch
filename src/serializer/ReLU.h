#pragma once
#include "../../include/nn/ReLU.h"
#include "Threshold.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class ReLU : public nn::ReLU<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                CHECK_AND_CAST(ReLU, Threshold, T)->unserialize(obj, mb);
            }
        };
    }
}
