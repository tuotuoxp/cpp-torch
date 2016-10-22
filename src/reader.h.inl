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

#define CHECK_AND_CAST(class_name, class_base_name, T) Cast<serializer::class_name<T,C>, serializer::class_base_name<T,C>, nn::class_name<T,C>, nn::class_base_name<T,C>>(this)


#include "serializer/BatchNormalization.h"
#include "serializer/Concat.h"
#include "serializer/Container.h"
#include "serializer/Decorator.h"
#include "serializer/DepthConcat.h"
#include "serializer/Inception.h"
#include "serializer/Linear.h"
#include "serializer/MulConstant.h"
#include "serializer/Normalize.h"
#include "serializer/ReLU.h"
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


template<typename T, bool C>
object_reader<T,C>::object_reader()
{
    addClass<cpptorch::serializer::BatchNormalization<T, C>>("nn.BatchNormalization");
    addClass<cpptorch::serializer::Concat<T, C>>("nn.Concat");
    addClass<cpptorch::serializer::Decorator<T, C>>("nn.Decorator");
    addClass<cpptorch::serializer::DepthConcat<T, C>>("nn.DepthConcat");
    addClass<cpptorch::serializer::Inception<T, C>>("nn.Inception");
    addClass<cpptorch::serializer::Linear<T, C>>("nn.Linear");
    addClass<cpptorch::serializer::MulConstant<T, C>>("nn.MulConstant");
    addClass<cpptorch::serializer::Normalize<T, C>>("nn.Normalize");
    addClass<cpptorch::serializer::ReLU<T, C>>("nn.ReLU");
    addClass<cpptorch::serializer::Reshape<T, C>>("nn.Reshape");
    addClass<cpptorch::serializer::Sequential<T, C>>("nn.Sequential");
    addClass<cpptorch::serializer::SpatialAveragePooling<T, C>>("nn.SpatialAveragePooling");
    addClass<cpptorch::serializer::SpatialBatchNormalization<T, C>>("nn.SpatialBatchNormalization");
    addClass<cpptorch::serializer::SpatialConvolution<T, C>>("nn.SpatialConvolution");
    addClass<cpptorch::serializer::SpatialConvolutionMM<T, C>>("nn.SpatialConvolutionMM");
    addClass<cpptorch::serializer::SpatialCrossMapLRN<T, C>>("nn.SpatialCrossMapLRN");
    addClass<cpptorch::serializer::SpatialLPPooling<T, C>>("nn.SpatialLPPooling");
    addClass<cpptorch::serializer::SpatialMaxPooling<T, C>>("nn.SpatialMaxPooling");
    addClass<cpptorch::serializer::SpatialReflectionPadding<T, C>>("nn.SpatialReflectionPadding");
    addClass<cpptorch::serializer::Sqrt<T, C>>("nn.Sqrt");
    addClass<cpptorch::serializer::Square<T, C>>("nn.Square");
    addClass<cpptorch::serializer::Threshold<T, C>>("nn.Threshold");
    addClass<cpptorch::serializer::View<T, C>>("nn.View");
}
    
template<typename T, bool C>
void object_reader<T,C>::build_storage(const cpptorch::object *obj, cpptorch::Storage<T,C> &storage)
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

template<typename T, bool C>
void object_reader<T,C>::build_from_size_storage(const cpptorch::object *obj, std::vector<long> &data)
{
    auto *obj_storage = obj->to_storage<long>();
    data.assign(obj_storage->storage_, obj_storage->storage_ + obj_storage->size_);
}


template<typename T, bool C>
cpptorch::Tensor<T,C> object_reader<T,C>::build_tensor(const cpptorch::object *obj)
{
    cpptorch::Tensor<T,C> out;
    const cpptorch::object_torch_tensor *obj_tensor = obj->to_tensor();
    if (obj_tensor->dimension_ > 0)
    {
        cpptorch::Storage<T,C> storage;
        build_storage(obj_tensor->data_.get(), storage);
        out.create(storage, (long)obj_tensor->storage_offset_, obj_tensor->dimension_, obj_tensor->size_, obj_tensor->stride_);
    }
    return std::move(out);
}

template<typename T, bool C>
std::shared_ptr<cpptorch::nn::Layer<T,C>> object_reader<T,C>::build_layer(const cpptorch::object *obj)
{
    const cpptorch::object_torch *obj_torch = obj->to_torch();
    auto it = layer_map_.find(obj_torch->index_);
    if (it == layer_map_.end())
    {
        auto factory = factory_.find(obj_torch->class_name_);
        if (factory != factory_.end())
        {
            std::shared_ptr<cpptorch::nn::Layer<T,C>> l((*factory->second)(obj_torch, this));
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

