#pragma once
#include "factory.h"

#include <assert.h>
#include <istream>
#include <vector>
#include <map>
#include <cmath>


class model_extractor;
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
    virtual void read(model_extractor &extractor, std::istream &is);

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

    void read(model_extractor &extractor, std::istream &is) override;

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

    void read(model_extractor &extractor, std::istream &is) override;

    long size_;
    T *storage_;
};



class model_extractor
{
public:

    model_extractor();

    std::shared_ptr<object> read_object(std::istream &is);



    static int read_int(std::istream &is)
    {
        int v;
        read_raw(is, &v, sizeof(int));
        return v;
    }

    static long read_long(std::istream &is)
    {
        long long v;
        read_raw(is, &v, sizeof(long long));
        return (long)v;
    }

    static double read_double(std::istream &is)
    {
        double v;
        read_raw(is, &v, sizeof(double));
        return v;
    }

    static std::string read_string_with_length(std::istream &is)
    {
        int len = read_int(is);
        std::string str(len, '\0');
        read_raw(is, &str[0], len);
        return str;
    }

    static void read_raw(std::istream &is, void *buf, size_t len)
    {
        is.read((char*)buf, len);
    }

    template<typename T>
    static T* read_array(std::istream &is, size_t count)
    {
        T *p = (T*)malloc(count * sizeof(T));
        is.read((char*)p, count * sizeof(T));
        return p;
    }


    // index -> object
    std::map<int, std::shared_ptr<object>> objects_;


    // torch object factory
    class_factory<object_torch> factory_;
};


inline object::operator bool() const
{
    assert(type_ == object_type_boolean);
    return (static_cast<const object_boolean*>(this))->val_;
}
inline object::operator int() const
{
    return (int)round((double)*this);
}
inline object::operator float() const
{
    return (float)(double)*this;
}
inline object::operator double() const
{
    assert(type_ == object_type_number);
    return (static_cast<const object_number*>(this))->num_;
}


inline const object_table* object::to_table() const
{
    assert(type_ == object_type_table);
    return static_cast<const object_table*>(this);
}
inline const object_torch* object::to_torch() const
{
    assert(type_ == object_type_torch);
    return static_cast<const object_torch*>(this);
}
inline const object_torch_tensor* object::to_tensor() const
{
    assert(type_ == object_type_torch_tensor);
    return static_cast<const object_torch_tensor*>(this);
}
template<typename T>
inline const object_torch_storage<T>* object::to_storage() const
{
    assert(type_ == object_type_torch_storage);
    return static_cast<const object_torch_storage<T>*>(this);
}


template<typename T>
void object_torch_storage<T>::read(model_extractor &extractor, std::istream &is)
{
    size_ = model_extractor::read_long(is);
    if (size_ > 0)
    {
        storage_ = model_extractor::read_array<T>(is, size_);
    }
}
