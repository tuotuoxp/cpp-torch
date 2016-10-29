#pragma once
#include "../../include/nn/View.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class View : public nn::View<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                this->num_elements_ = *obj_tbl->get("numElements");
                mb->build_from_size_storage(obj_tbl->get("size"), this->size_);
                const object *num_input_dims = obj_tbl->get("numInputDims");
                this->num_input_dims_ = num_input_dims ? *num_input_dims : -1;
            }
        };
    }
}
