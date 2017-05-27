#pragma once
#include "../include/builder.h"
#include "reader.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::read_tensor(const cpptorch::object *obj)
{
    object_reader<T, F> mb;
    return mb.build_tensor(obj);
}

template<typename T, GPUFlag F>
std::shared_ptr<cpptorch::nn::Layer<T, F>> cpptorch::read_net(const cpptorch::object *obj)
{
    object_reader<T, F> mb;
    return std::static_pointer_cast<cpptorch::nn::Layer<T, F>>(mb.build_layer(obj));
}
