#pragma once
#include "../../include/nn/Add.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Add : public nn::Add<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                if(obj_tbl->get("scalar"))
                    this->scalar_ = obj_tbl->get("scalar");
                else
                    this->scalar_ = false;
                this->bias_ = mb->build_tensor(obj_tbl->get("bias"));
                this->gradBias_ = mb->build_tensor(obj_tbl->get("gradBias"));
                this->_ones_ = mb->build_tensor(obj_tbl->get("_ones"));
            }
        };
    }
}
