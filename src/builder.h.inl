#pragma once
#include "../include/builder.h"
#include "reader.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::read_tensor(const cpptorch::object *obj)
{
    object_reader<TTensor> mb;
    return mb.build_tensor(obj);
}

template<class TTensor>
std::shared_ptr<cpptorch::nn::Layer<TTensor>> cpptorch::read_net(const cpptorch::object *obj)
{
    object_reader<TTensor> mb;
    return std::static_pointer_cast<cpptorch::nn::Layer<TTensor>>(mb.build_layer(obj));
}
