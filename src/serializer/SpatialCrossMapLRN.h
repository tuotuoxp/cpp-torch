#pragma once
#include "../nn/SpatialCrossMapLRN.h"


namespace serializer
{
    template<class TTensor>
    class SpatialCrossMapLRN : public nn::SpatialCrossMapLRN<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            const object_table *obj_tbl = obj->data_->to_table();
            this->size_ = *obj_tbl->get("size");
            this->alpha_ = *obj_tbl->get("alpha");
            this->beta_ = *obj_tbl->get("beta");
            this->k_ = *obj_tbl->get("k");
        }
    };
}
