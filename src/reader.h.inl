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

#define CHECK_AND_CAST(class_name, class_base_name, T) Cast<serializer::class_name<T>, serializer::class_base_name<T>, nn::class_name<T>, nn::class_base_name<T>>(this)


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


template<class TTensor>
object_reader<TTensor>::object_reader()
{
    CREATE_BUILDER("nn.BatchNormalization", cpptorch::serializer::BatchNormalization<TTensor>);
    CREATE_BUILDER("nn.Concat", cpptorch::serializer::Concat<TTensor>);
    CREATE_BUILDER("nn.Decorator", cpptorch::serializer::Decorator<TTensor>);
    CREATE_BUILDER("nn.DepthConcat", cpptorch::serializer::DepthConcat<TTensor>);
    CREATE_BUILDER("nn.Inception", cpptorch::serializer::Inception<TTensor>);
    CREATE_BUILDER("nn.Linear", cpptorch::serializer::Linear<TTensor>);
    CREATE_BUILDER("nn.MulConstant", cpptorch::serializer::MulConstant<TTensor>);
    CREATE_BUILDER("nn.Normalize", cpptorch::serializer::Normalize<TTensor>);
    CREATE_BUILDER("nn.ReLU", cpptorch::serializer::ReLU<TTensor>);
    CREATE_BUILDER("nn.Reshape", cpptorch::serializer::Reshape<TTensor>);
    CREATE_BUILDER("nn.Sequential", cpptorch::serializer::Sequential<TTensor>);
    CREATE_BUILDER("nn.SpatialAveragePooling", cpptorch::serializer::SpatialAveragePooling<TTensor>);
    CREATE_BUILDER("nn.SpatialBatchNormalization", cpptorch::serializer::SpatialBatchNormalization<TTensor>);
    CREATE_BUILDER("nn.SpatialConvolution", cpptorch::serializer::SpatialConvolution<TTensor>);
    CREATE_BUILDER("nn.SpatialConvolutionMM", cpptorch::serializer::SpatialConvolutionMM<TTensor>);
    CREATE_BUILDER("nn.SpatialCrossMapLRN", cpptorch::serializer::SpatialCrossMapLRN<TTensor>);
    CREATE_BUILDER("nn.SpatialLPPooling", cpptorch::serializer::SpatialLPPooling<TTensor>);
    CREATE_BUILDER("nn.SpatialMaxPooling", cpptorch::serializer::SpatialMaxPooling<TTensor>);
    CREATE_BUILDER("nn.SpatialReflectionPadding", cpptorch::serializer::SpatialReflectionPadding<TTensor>);
    CREATE_BUILDER("nn.Sqrt", cpptorch::serializer::Sqrt<TTensor>);
    CREATE_BUILDER("nn.Square", cpptorch::serializer::Square<TTensor>);
    CREATE_BUILDER("nn.Threshold", cpptorch::serializer::Threshold<TTensor>);
    CREATE_BUILDER("nn.View", cpptorch::serializer::View<TTensor>);
}
    
template<class TTensor>
void object_reader<TTensor>::build_storage(const cpptorch::object *obj, cpptorch::Storage<typename TTensor::Storage> &storage)
{
    auto obj_storage = const_cast<cpptorch::object_torch_storage<typename TTensor::Storage::Base>*>(obj->to_storage<typename TTensor::Storage::Base>());
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

template<class TTensor>
void object_reader<TTensor>::build_from_size_storage(const cpptorch::object *obj, std::vector<long> &data)
{
    auto *obj_storage = obj->to_storage<long>();
    data.assign(obj_storage->storage_, obj_storage->storage_ + obj_storage->size_);
}


template<class TTensor>
cpptorch::Tensor<TTensor> object_reader<TTensor>::build_tensor(const cpptorch::object *obj)
{
    cpptorch::Tensor<TTensor> out;
    const cpptorch::object_torch_tensor *obj_tensor = obj->to_tensor();
    if (obj_tensor->dimension_ > 0)
    {
        cpptorch::Storage<typename TTensor::Storage> storage;
        build_storage(obj_tensor->data_.get(), storage);
        cpptorch::Storage<typename TTensor::SizeStorage> size, stride;
        size.unserialze(obj_tensor->size_, obj_tensor->dimension_, false);
        stride.unserialze(obj_tensor->stride_, obj_tensor->dimension_, false);
        out.create(storage, (long)obj_tensor->storage_offset_, size, stride);
    }
    return std::move(out);
}

template<class TTensor>
std::shared_ptr<cpptorch::nn::Layer<TTensor>> object_reader<TTensor>::build_layer(const cpptorch::object *obj)
{
    const cpptorch::object_torch *obj_torch = obj->to_torch();
    auto it = layer_map_.find(obj_torch->index_);
    if (it == layer_map_.end())
    {
        auto factory = factory_.find(obj_torch->class_name_);
        if (factory != factory_.end())
        {
            std::shared_ptr<cpptorch::nn::Layer<TTensor>> l((*factory->second)(obj_torch, this));
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

