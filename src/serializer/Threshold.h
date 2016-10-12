#pragma once
#include "../../include/nn/Threshold.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T>
        class Threshold : public nn::Threshold<T>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->threshold_ = *obj_tbl->get("threshold");
                this->val_ = *obj_tbl->get("val");
                this->inplace_ = *obj_tbl->get("inplace");
            }
        };
    }
}
