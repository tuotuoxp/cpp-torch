#pragma once
#include "../../include/nn/LogSoftMax.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class LogSoftMax : public nn::LogSoftMax<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {

            }
        };
    }
}
