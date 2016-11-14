#pragma once
#include "torch/Tensor.h"
#include "nn/Layer.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <assert.h>
#include <math.h>


namespace cpptorch
{
    class object_table;
    class object_torch;
    class object_torch_tensor;
    template<typename T> class object_torch_storage;

    enum object_type
    {
        object_type_nil,
        object_type_number,
        object_type_string,
        object_type_boolean,
        object_type_table,
        object_type_torch,
        object_type_torch_tensor,
        object_type_torch_storage,
    };


    class object
    {
    public:
        object(object_type t = object_type_nil) : type_(t) {}

        virtual ~object() {}

        operator bool() const;
        operator int() const;
        operator long() const;
        operator float() const;
        operator double() const;

        inline const object_table* to_table() const;
        inline const object_torch* to_torch() const;
        inline const object_torch_tensor* to_tensor() const;
        template<typename T>
        inline const object_torch_storage<T>* to_storage() const;

        object_type type_;
    };

    class object_number : public object
    {
    public:
        object_number() : object(object_type_number) {}

        double num_;
    };

    class object_boolean : public object
    {
    public:
        object_boolean() : object(object_type_boolean) {}

        bool val_;
    };

    class object_string : public object
    {
    public:
        object_string() : object(object_type_string) {}

        std::string str_;
    };

    class object_table : public object
    {
    public:
        object_table() : object(object_type_table) {}

        const object* get(const std::string &key) const
        {
            auto it = kv_.find(key);
            return it == kv_.end() ? nullptr : it->second.get();
        }
        const object* get(size_t key) const
        {
            return key < array_.size() ? array_[key].get() : nullptr;
        }

        std::map<std::string, std::shared_ptr<object>> kv_;
        std::vector<std::shared_ptr<object>> array_;
    };

    class object_torch : public object
    {
    public:
        object_torch(object_type t = object_type_torch) : object(t) {}

        int version_;
        int index_;
        std::string class_name_;
        std::shared_ptr<object> data_;
    };

    class object_torch_tensor : public object_torch
    {
    public:
        object_torch_tensor() : object_torch(object_type_torch_tensor), size_(nullptr), stride_(nullptr) {}
        ~object_torch_tensor()
        {
            if (size_)
            {
                free(size_);
                size_ = nullptr;
            }
            if (stride_)
            {
                free(stride_);
                stride_ = nullptr;
            }
        }

        int dimension_;
        long *size_, *stride_;
        long storage_offset_;
    };

    template<typename T>
    class object_torch_storage : public object_torch
    {
    public:
        object_torch_storage() : object_torch(object_type_torch_storage), storage_(nullptr) {}
        ~object_torch_storage()
        {
            if (storage_)
            {
                free(storage_);
                storage_ = nullptr;
            }
        }

        long size_;
        T *storage_;
    };


    template <typename T, GPUFlag F = GPU_None>
    class layer_creator
    {
    public:
        virtual std::vector<std::string> register_layers() = 0;
        virtual std::shared_ptr<nn::Layer<T, F>> create_layer(const std::string &layer_name, const cpptorch::object_torch *torch_obj) = 0;

        void *context_;
    };


    // load module utils
    API std::shared_ptr<object> load(std::istream &is);

    template<typename T>
    API Tensor<T, GPU_None> read_tensor(const object *obj);
    template<typename T>
    API std::shared_ptr<nn::Layer<T, GPU_None>> read_net(const object *obj, layer_creator<T, GPU_None> *creator = nullptr);
}


inline cpptorch::object::operator bool() const
{
    assert(type_ == object_type_boolean);
    return (static_cast<const object_boolean*>(this))->val_;
}
inline cpptorch::object::operator int() const
{
    return (int)round((double)*this);
}
inline cpptorch::object::operator long() const
{
    return (long)round((double)*this);
}
inline cpptorch::object::operator float() const
{
    return (float)(double)*this;
}
inline cpptorch::object::operator double() const
{
    assert(type_ == object_type_number);
    return (static_cast<const object_number*>(this))->num_;
}


inline const cpptorch::object_table* cpptorch::object::to_table() const
{
    assert(type_ == object_type_table);
    return static_cast<const cpptorch::object_table*>(this);
}
inline const cpptorch::object_torch* cpptorch::object::to_torch() const
{
    assert(type_ == object_type_torch);
    return static_cast<const cpptorch::object_torch*>(this);
}
inline const cpptorch::object_torch_tensor* cpptorch::object::to_tensor() const
{
    assert(type_ == object_type_torch_tensor);
    return static_cast<const cpptorch::object_torch_tensor*>(this);
}
template<typename T>
inline const cpptorch::object_torch_storage<T>* cpptorch::object::to_storage() const
{
    assert(type_ == object_type_torch_storage);
    return static_cast<const cpptorch::object_torch_storage<T>*>(this);
}
