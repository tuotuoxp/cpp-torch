#pragma once
#include "../../include/nn/Concat.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, bool C>
        class Concat : public nn::Concat<T,C>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T,C> *mb)
            {
                CHECK_AND_CAST(Concat, Container, T)->unserialize(obj, mb);
                const object_table *obj_tbl = obj->data_->to_table();
                this->dimension_ = (int)*obj_tbl->get("dimension") - 1;
            }
        };
    }
}
