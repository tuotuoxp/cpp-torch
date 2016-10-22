#pragma once
#include "../../include/nn/Sqrt.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class Sqrt : public nn::Sqrt<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->eps_ = *obj_tbl->get("eps");
            }
        };
    }
}
