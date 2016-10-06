#pragma once
#include "../nn/ReLU.h"
#include "Threshold.h"


namespace serializer
{
    template<class TTensor>
    class ReLU : public nn::ReLU<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            CHECK_AND_CAST(ReLU, Threshold, TTensor)->unserialize(obj, mb);
        }
    };
}
