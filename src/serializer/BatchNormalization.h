#pragma once
#include "../nn/BatchNormalization.h"


namespace serializer
{
    template<class TTensor>
    class BatchNormalization : public nn::BatchNormalization<TTensor>
    {
    public:
        void unserialize(const object_torch *obj, model_builder<TTensor> *mb)
        {
            const object_table *obj_tbl = obj->data_->to_table();
            this->weight_ = mb->build_tensor(obj_tbl->get("weight"));
            this->bias_ = mb->build_tensor(obj_tbl->get("bias"));
            this->running_mean_ = mb->build_tensor(obj_tbl->get("running_mean"));
            this->momentum_ = *obj_tbl->get("momentum");
            this->eps_ = *obj_tbl->get("eps");
            this->train_ = false;
            if (obj->version_ < 2)
            {
                nn::Tensor<TTensor> running_std = mb->build_tensor(obj_tbl->get("running_std"));
                this->running_var_ = (running_std ^ -2.0f) + (typename TTensor::Storage::StorageBase) - this->eps_;
            }
            else
            {
                this->running_var_ = mb->build_tensor(obj_tbl->get("running_var"));
            }
        }
    };
}
