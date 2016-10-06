#pragma once
#include "../nn/Threshold.h"


namespace serializer
{
    template<class TTensor>
    class Threshold : public nn::Threshold<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            const object_table *obj_tbl = obj->data_->to_table();
            this->threshold_ = *obj_tbl->get("threshold");
            this->val_ = *obj_tbl->get("val");
            this->inplace_ = *obj_tbl->get("inplace");
        }
    };
}
