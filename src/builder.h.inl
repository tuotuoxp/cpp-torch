#pragma once
#include "../include/builder.h"
#include "reader.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::layer_creator<T, F>::read_tensor(const cpptorch::object *obj)
{
    return ((object_reader<T, F>*)context_)->build_tensor(obj);
}


//////////////////////////////////////////////////////////////////////////


template<typename T>
cpptorch::Tensor<T, GPU_None> cpptorch::read_tensor(const cpptorch::object *obj)
{
    object_reader<T, GPU_None> mb;
    return mb.build_tensor(obj);
}

template<typename T>
std::shared_ptr<cpptorch::nn::Layer<T, GPU_None>> cpptorch::read_net(const cpptorch::object *obj,
    cpptorch::layer_creator<T, GPU_None> *creator)
{
    object_reader<T, GPU_None> mb(creator);
    return std::static_pointer_cast<cpptorch::nn::Layer<T, GPU_None>>(mb.build_layer(obj));
}
