#pragma once
#include "../../include/nn/Linear.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Linear : public nn::Linear<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->weight_ = mb->build_tensor(obj_tbl->get("weight"));
                this->bias_ = mb->build_tensor(obj_tbl->get("bias"));
            }
        };
    }
}
