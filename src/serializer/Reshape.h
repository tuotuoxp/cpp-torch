#pragma once
#include "../nn/Reshape.h"


namespace serializer
{
    template<class TTensor>
    class Reshape : public nn::Reshape<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            const object_table *obj_tbl = obj->data_->to_table();
            this->nelement_ = *obj_tbl->get("nelement");
            mb->build_from_size_storage(obj_tbl->get("size"), this->size_);
            mb->build_from_size_storage(obj_tbl->get("batchsize"), this->batchsize_);
            const object *batch_mode = obj_tbl->get("batchMode");
            this->batch_mode_ = batch_mode ? *batch_mode : false;
        }
    };
}
