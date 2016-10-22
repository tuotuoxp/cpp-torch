#pragma once
#include "../../include/nn/Normalize.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class Normalize : public nn::Normalize<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->p_ = *obj_tbl->get("p");
                this->eps_ = *obj_tbl->get("eps");
            }
        };
    }
}
