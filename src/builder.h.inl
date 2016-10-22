#pragma once
#include "../include/builder.h"
#include "reader.h"


template<typename T>
cpptorch::Tensor<T, false> cpptorch::read_tensor(const cpptorch::object *obj)
{
    object_reader<T, false> mb;
    return mb.build_tensor(obj);
}

template<typename T>
std::shared_ptr<cpptorch::nn::Layer<T, false>> cpptorch::read_net(const cpptorch::object *obj)
{
    object_reader<T, false> mb;
    return std::static_pointer_cast<cpptorch::nn::Layer<T, false>>(mb.build_layer(obj));
}
