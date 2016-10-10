#pragma once
#include "../../include/nn/Concat.h"
#include "Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class Concat : public nn::Concat<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                CHECK_AND_CAST(Concat, Container, TTensor)->unserialize(obj, mb);
                const object_table *obj_tbl = obj->data_->to_table();
                this->dimension_ = (int)*obj_tbl->get("dimension") - 1;
            }
        };
    }
}
