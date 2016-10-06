#pragma once
#include "../nn/Sqrt.h"


namespace serializer
{
    template<class TTensor>
    class Sqrt : public nn::Sqrt<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            const object_table *obj_tbl = obj->data_->to_table();
            this->eps_ = *obj_tbl->get("eps");
        }
    };
}
