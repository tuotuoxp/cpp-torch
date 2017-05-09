#pragma once
#include "reader.h"


template<class TSerializer, class TSerializerBase, class TNN, class TNNBase>
TSerializerBase* Cast(TSerializer *c)
{
    static_assert(std::is_base_of<TNN, TSerializer>::value, "serializer class should derived from the nn class");
    static_assert(std::is_base_of<TNNBase, TSerializerBase>::value, "serializer class should derived from the nn class");
    static_assert(std::is_base_of<TNNBase, TNN>::value, "only casting to base class allowd");
    return (TSerializerBase*)c;
}

#define CHECK_AND_CAST(class_name, class_base_name, T) Cast<serializer::class_name<T, F>, serializer::class_base_name<T, F>, nn::class_name<T, F>, nn::class_base_name<T, F>>(this)


#include "serializer/BatchNormalization.h"
#include "serializer/Concat.h"
#include "serializer/Container.h"
#include "serializer/Decorator.h"
#include "serializer/DepthConcat.h"
#include "serializer/Inception.h"
#include "serializer/Linear.h"
#include "serializer/Add.h"
#include "serializer/MulConstant.h"
#include "serializer/Normalize.h"
#include "serializer/ReLU.h"
#include "serializer/SoftMax.h"
#include "serializer/LogSoftMax.h"
#include "serializer/Reshape.h"
#include "serializer/Sequential.h"
#include "serializer/SpatialAveragePooling.h"
#include "serializer/SpatialBatchNormalization.h"
#include "serializer/SpatialConvolution.h"
#include "serializer/SpatialConvolutionMM.h"
#include "serializer/SpatialCrossMapLRN.h"
#include "serializer/SpatialLPPooling.h"
#include "serializer/SpatialMaxPooling.h"
#include "serializer/SpatialReflectionPadding.h"
#include "serializer/Sqrt.h"
#include "serializer/Square.h"
#include "serializer/Threshold.h"
#include "serializer/View.h"


template<typename T, GPUFlag F>
object_reader<T, F>::object_reader()
{
    addClass<cpptorch::serializer::BatchNormalization<T, F>>("nn.BatchNormalization");
    addClass<cpptorch::serializer::Concat<T, F>>("nn.Concat");
    addClass<cpptorch::serializer::Decorator<T, F>>("nn.Decorator");
    addClass<cpptorch::serializer::DepthConcat<T, F>>("nn.DepthConcat");
    addClass<cpptorch::serializer::Inception<T, F>>("nn.Inception");
    addClass<cpptorch::serializer::Linear<T, F>>("nn.Linear");
    addClass<cpptorch::serializer::Add<T, F>>("nn.Add");
    addClass<cpptorch::serializer::MulConstant<T, F>>("nn.MulConstant");
    addClass<cpptorch::serializer::Normalize<T, F>>("nn.Normalize");
    addClass<cpptorch::serializer::ReLU<T, F>>("nn.ReLU");
    addClass<cpptorch::serializer::SoftMax<T, F>>("nn.SoftMax");
    addClass<cpptorch::serializer::LogSoftMax<T, F>>("nn.LogSoftMax");
    addClass<cpptorch::serializer::Reshape<T, F>>("nn.Reshape");
    addClass<cpptorch::serializer::Sequential<T, F>>("nn.Sequential");
    addClass<cpptorch::serializer::SpatialAveragePooling<T, F>>("nn.SpatialAveragePooling");
    addClass<cpptorch::serializer::SpatialBatchNormalization<T, F>>("nn.SpatialBatchNormalization");
    addClass<cpptorch::serializer::SpatialConvolution<T, F>>("nn.SpatialConvolution");
    addClass<cpptorch::serializer::SpatialConvolutionMM<T, F>>("nn.SpatialConvolutionMM");
    addClass<cpptorch::serializer::SpatialCrossMapLRN<T, F>>("nn.SpatialCrossMapLRN");
    addClass<cpptorch::serializer::SpatialLPPooling<T, F>>("nn.SpatialLPPooling");
    addClass<cpptorch::serializer::SpatialMaxPooling<T, F>>("nn.SpatialMaxPooling");
    addClass<cpptorch::serializer::SpatialReflectionPadding<T, F>>("nn.SpatialReflectionPadding");
    addClass<cpptorch::serializer::Sqrt<T, F>>("nn.Sqrt");
    addClass<cpptorch::serializer::Square<T, F>>("nn.Square");
    addClass<cpptorch::serializer::Threshold<T, F>>("nn.Threshold");
    addClass<cpptorch::serializer::View<T, F>>("nn.View");
}

template<typename T, GPUFlag F>
void object_reader<T, F>::build_storage(const cpptorch::object *obj, cpptorch::Storage<T, F> &storage)
{
    auto obj_storage = const_cast<cpptorch::object_torch_storage<T>*>(obj->to_storage<T>());
    auto it = storage_map_.find(obj_storage->index_);
    if (it == storage_map_.end())
    {
        storage.unserialze(obj_storage->storage_, obj_storage->size_);
        // move the ownership of data to cpptorch::Storage
        obj_storage->storage_ = nullptr;
        storage_map_.insert(std::make_pair(obj_storage->index_, storage));
    }
    else
    {
        storage = it->second;
    }
}

template<typename T, GPUFlag F>
void object_reader<T, F>::build_from_size_storage(const cpptorch::object *obj, std::vector<long> &data)
{
    auto *obj_storage = obj->to_storage<long>();
    data.assign(obj_storage->storage_, obj_storage->storage_ + obj_storage->size_);
}


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> object_reader<T, F>::build_tensor(const cpptorch::object *obj)
{
    cpptorch::Tensor<T, F> out;
    const cpptorch::object_torch_tensor *obj_tensor = obj->to_tensor();
    if (obj_tensor->dimension_ > 0)
    {
        cpptorch::Storage<T, F> storage;
        build_storage(obj_tensor->data_.get(), storage);
        out.create(storage, (long)obj_tensor->storage_offset_, obj_tensor->dimension_, obj_tensor->size_, obj_tensor->stride_);
    }
    return std::move(out);
}

template<typename T, GPUFlag F>
std::shared_ptr<cpptorch::nn::Layer<T, F>> object_reader<T, F>::build_layer(const cpptorch::object *obj)
{
    const cpptorch::object_torch *obj_torch = obj->to_torch();
    auto it = layer_map_.find(obj_torch->index_);
    if (it == layer_map_.end())
    {
        auto factory = factory_.find(obj_torch->class_name_);
        if (factory != factory_.end())
        {
            std::shared_ptr<cpptorch::nn::Layer<T, F>> l((*factory->second)(obj_torch, this));
            layer_map_.insert(std::make_pair(obj_torch->index_, l));
            return l;
        }
        assert(0);
        return nullptr;
    }
    else
    {
        return it->second;
    }
}
