#pragma once
#include "../../include/nn/SpatialReflectionPadding.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class SpatialReflectionPadding : public nn::SpatialReflectionPadding<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->pad_l_ = *obj_tbl->get("pad_l");
                this->pad_r_ = *obj_tbl->get("pad_r");
                this->pad_t_ = *obj_tbl->get("pad_t");
                this->pad_b_ = *obj_tbl->get("pad_b");
            }
        };
    }
}
