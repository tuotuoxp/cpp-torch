#pragma once
#include "nn/Tensor.h"

#include <map>


class object;
class object_torch;

template<class TTensor>
class model_builder
{
public:
    model_builder();


    void build_storage(const object *obj, nn::Storage<typename TTensor::Storage> &storage);
    void build_from_size_storage(const object *obj, std::vector<long> &data);
    nn::Tensor<TTensor> build_tensor(const object *obj);
    std::shared_ptr<nn::Layer<TTensor>> build_layer(const object *obj);

private:

    // objects can be referenced more than once, when one object is refereneced the second time,
    // we cannot create a new nn class, we should copy the first one,
    // so here we keep a map for each nn object we created

    // Layer cannot be copied, only referenced by pointer allowed
    std::map<int, std::shared_ptr<nn::Layer<TTensor>>> layer_map_;

    // Storage can be copied, because THStorage is self-counted
    std::map<int, nn::Storage<typename TTensor::Storage>> storage_map_;


    class class_factory_base
    {
    public:
        virtual nn::Layer<TTensor>* operator()(const object_torch *obj, model_builder<TTensor> *mb) const = 0;
    };

    template <class T>
    class class_factory_impl : public class_factory_base
    {
    public:
        virtual nn::Layer<TTensor>* operator()(const object_torch *obj, model_builder<TTensor> *mb) const
        {
            T *t = new T();
            t->unserialize(obj, mb);
            return static_cast<nn::Layer<TTensor>*>(t);
        }
    };
    // class name to builder
    std::map<std::string, std::shared_ptr<class_factory_base>> factory_;
};
