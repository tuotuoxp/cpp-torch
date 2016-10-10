#pragma once
#include "../../include/nn/Sqrt.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class Sqrt : public nn::Sqrt<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->eps_ = *obj_tbl->get("eps");
            }
        };
    }
}
