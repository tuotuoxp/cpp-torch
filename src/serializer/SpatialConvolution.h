#pragma once
#include "../../include/nn/SpatialConvolution.h"


namespace cpptorch
{
    namespace serializer
    {
        template<class TTensor>
        class SpatialConvolution : public nn::SpatialConvolution<TTensor>
        {
        public:
            void unserialize(const object_torch *obj, object_reader<TTensor> *mb)
            {
                const object_table *obj_tbl = obj->data_->to_table();

                this->weight_ = mb->build_tensor(obj_tbl->get("weight"));
                this->bias_ = mb->build_tensor(obj_tbl->get("bias"));

                this->kW_ = *obj_tbl->get("kW");
                this->kH_ = *obj_tbl->get("kH");
                this->dW_ = *obj_tbl->get("dW");
                this->dH_ = *obj_tbl->get("dH");

                const object *padding_obj = obj_tbl->get("padding");
                if (padding_obj)
                {
                    // backward compatibility
                    this->padH_ = this->padW_ = *padding_obj;
                }
                else
                {
                    this->padW_ = *obj_tbl->get("padW");
                    this->padH_ = *obj_tbl->get("padH");
                }

                if (this->weight_.dim() == 2)
                {
                    int output_plane = *obj_tbl->get("nOutputPlane");
                    int input_plane = *obj_tbl->get("nInputPlane");
                    this->weight_ = this->weight_.view({ output_plane, input_plane, this->kH_, this->kW_ });
                }
            }
        };
    }
}
