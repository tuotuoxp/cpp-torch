#pragma once
#include "../include/torch/Tensor.h"
#include "../include/nn/Layer.h"
#include "../include/builder.h"

#include <map>


#define CREATE_BUILDER(name, T)     this->factory_.insert(std::make_pair(name, std::shared_ptr<class_factory_base>(new class_factory_impl<T>())))


template<typename T>
class object_reader
{
public:
    object_reader();


    void build_storage(const cpptorch::object *obj, cpptorch::Storage<T> &storage);
    void build_from_size_storage(const cpptorch::object *obj, std::vector<long> &data);
    cpptorch::Tensor<T> build_tensor(const cpptorch::object *obj);
    std::shared_ptr<cpptorch::nn::Layer<T>> build_layer(const cpptorch::object *obj);


protected:

    class class_factory_base
    {
    public:
        virtual cpptorch::nn::Layer<T>* operator()(const cpptorch::object_torch *obj, object_reader<T> *mb) const = 0;
    };

    template <class TNN>
    class class_factory_impl : public class_factory_base
    {
    public:
        virtual cpptorch::nn::Layer<T>* operator()(const cpptorch::object_torch *obj, object_reader<T> *mb) const
        {
            TNN *t = new TNN();
            t->unserialize(obj, mb);
            return static_cast<cpptorch::nn::Layer<T>*>(t);
        }
    };
    // class name to builder
    std::map<std::string, std::shared_ptr<class_factory_base>> factory_;


private:

    // objects can be referenced more than once, when one object is refereneced the second time,
    // we cannot create a new nn class, we should copy the first one,
    // so here we keep a map for each nn object we created

    // Layer cannot be copied, only referenced by pointer allowed
    std::map<int, std::shared_ptr<cpptorch::nn::Layer<T>>> layer_map_;

    // Storage can be copied, because TH is self-counted
    std::map<int, cpptorch::Storage<T>> storage_map_;
};
