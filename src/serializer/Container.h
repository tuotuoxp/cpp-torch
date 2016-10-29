#pragma once
#include "../../include/nn/Container.h"


namespace cpptorch
{
    namespace serializer
    {
        template<typename T, GPUFlag F>
        class Container : public nn::Container<T, F>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<T, F> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();
                const object_table *obj_modules = obj_tbl->get("modules")->to_table();
                for (auto &it_obj : obj_modules->array_)
                {
                    this->modules_.push_back(std::static_pointer_cast<nn::Layer<T, F>>(mb->build_layer(it_obj.get())));
                }
            }
        };
    }
}
