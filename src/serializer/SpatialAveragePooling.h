#pragma once
#include "../../include/nn/SpatialAveragePooling.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class SpatialAveragePooling : public nn::SpatialAveragePooling<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->kW_ = *obj_tbl->get("kW");
                this->kH_ = *obj_tbl->get("kH");
                this->dW_ = *obj_tbl->get("dW");
                this->dH_ = *obj_tbl->get("dH");

                const object *ceil_mode = obj_tbl->get("ceil_mode");
                if (ceil_mode == nullptr)
                {
                    // backward compatible
                    this->ceil_mode_ = false;
                    this->count_include_pad_ = true;
                    this->padW_ = 0;
                    this->padH_ = 0;
                }
                else
                {
                    this->ceil_mode_ = *ceil_mode;
                    this->count_include_pad_ = *obj_tbl->get("count_include_pad");
                    this->padW_ = *obj_tbl->get("padW");
                    this->padH_ = *obj_tbl->get("padH");
                }

                const object *divide = obj_tbl->get("divide");
                this->divide_ = divide ? *divide : false;
            }
        };
    }
}
