#pragma once
#include "../nn/SpatialReflectionPadding.h"


namespace serializer
{
    template<class TTensor>
    class SpatialReflectionPadding : public nn::SpatialReflectionPadding<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            const object_table *obj_tbl = obj->data_->to_table();
            this->pad_l_ = *obj_tbl->get("pad_l");
            this->pad_r_ = *obj_tbl->get("pad_r");
            this->pad_t_ = *obj_tbl->get("pad_t");
            this->pad_b_ = *obj_tbl->get("pad_b");
        }
    };
}
