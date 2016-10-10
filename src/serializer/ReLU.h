#pragma once
#include "../../include/nn/ReLU.h"
#include "Threshold.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class ReLU : public nn::ReLU<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(ReLU, Threshold, TTensor)->unserialize(obj, mb);
            }
        };
    }
}
