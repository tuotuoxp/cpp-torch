#include "loader.h"

#include <string>
#include <limits>


#define TYPE_NIL                     0
#define TYPE_NUMBER                  1
#define TYPE_STRING                  2
#define TYPE_TABLE                   3
#define TYPE_TORCH                   4
#define TYPE_BOOLEAN                 5
#define TYPE_FUNCTION                6
#define TYPE_RECUR_FUNCTION          8
#define LEGACY_TYPE_RECUR_FUNCTION   7


template<>
long* object_loader::read_array<long>(std::istream &is, size_t count)
{
    long *p = (long*)malloc(count * sizeof(long long));
    is.read((char*)p, count * sizeof(long long));

    // in windows, sizeof(long) != sizeof(long long)
    if (sizeof(long) != sizeof(long long))
    {
        long long *pll = (long long*)p;
        for (size_t i = 0; i < count; i++)
        {
            p[i] = (long)pll[i];
        }
        p = (long*)realloc(p, count * sizeof(long));
    }
    return p;
}


//////////////////////////////////////////////////////////////////////////


class object_torch_reader : public cpptorch::object_torch
{
public:
    void read(object_loader &extractor, std::istream &is)
    {
        data_ = extractor.read_object(is);
    }
};

class object_torch_tensor_reader : public cpptorch::object_torch_tensor
{
public:
    void read(object_loader &extractor, std::istream &is)
    {
        dimension_ = object_loader::read_int(is);
        if (dimension_ > 0)
        {
            size_ = object_loader::read_array<long>(is, dimension_);
            stride_ = object_loader::read_array<long>(is, dimension_);
        }
        storage_offset_ = object_loader::read_long(is) - 1;
        data_ = extractor.read_object(is);
    }
};

template <typename T>
class object_torch_storage_reader : public cpptorch::object_torch_storage<T>
{
public:
    void read(object_loader &extractor, std::istream &is)
    {
        this->size_ = object_loader::read_long(is);
        if (this->size_ > 0)
        {
            this->storage_ = object_loader::read_array<T>(is, this->size_);
        }
    }
};


//////////////////////////////////////////////////////////////////////////


#define CREATE_BUILDER(name, T)     this->factory_.insert(std::make_pair(name, std::shared_ptr<class_factory_base>(new class_factory_impl<T, F>())))


object_loader::object_loader()
{
    addClass<object_torch_tensor_reader>("torch.LongTensor");
    addClass<object_torch_tensor_reader>("torch.FloatTensor");
    addClass<object_torch_tensor_reader>("torch.DoubleTensor");
    addClass<object_torch_storage_reader<long>>("torch.LongStorage");
    addClass<object_torch_storage_reader<float>>("torch.FloatStorage");
    addClass<object_torch_storage_reader<double>>("torch.DoubleStorage");
}

std::shared_ptr<cpptorch::object> object_loader::read_object(std::istream &is)
{
    int typeidx = read_int(is);

    switch (typeidx)
    {
    case TYPE_NIL:
        return std::make_shared<cpptorch::object>();
    case TYPE_NUMBER:
    {
        std::shared_ptr<cpptorch::object_number> on = std::make_shared<cpptorch::object_number>();
        on->num_ = read_double(is);
        return on;
    }
    case TYPE_STRING:
    {
        std::shared_ptr<cpptorch::object_string> os = std::make_shared<cpptorch::object_string>();
        os->str_ = read_string_with_length(is);
        return os;
    }
    case TYPE_BOOLEAN:
    {
        std::shared_ptr<cpptorch::object_boolean> ob = std::make_shared<cpptorch::object_boolean>();
        ob->val_ = read_int(is) == 1;
        return ob;
    }

    case TYPE_TABLE:
    case TYPE_TORCH:
    case TYPE_RECUR_FUNCTION:
    case LEGACY_TYPE_RECUR_FUNCTION:
        int index = read_int(is);

        auto it = objects_.find(index);
        if (it != objects_.end())
        {
            return it->second;
        }

        if (typeidx == TYPE_TORCH)
        {
            std::string version = read_string_with_length(is);
            std::string class_name = version;
            int version_number = 0;
            if (version.length() > 2 && version.substr(0, 2) == "V ")
            {
                version = version.substr(2);
                char *p;
                strtol(version.c_str(), &p, 10);
                if (*p == 0)
                {
                    version_number = atoi(version.c_str());
                    class_name = read_string_with_length(is);
                }
            }

            std::shared_ptr<cpptorch::object_torch> obj_th;
            auto factory = factory_.find(class_name);
            if (factory != factory_.end())
            {
                obj_th = std::shared_ptr<cpptorch::object_torch>(factory->second->create(*this, is));
            }
            else
            {
                obj_th = std::make_shared<cpptorch::object_torch>();
                ((object_torch_reader*)(obj_th.get()))->read(*this, is);
            }
            obj_th->version_ = version_number;
            obj_th->index_ = index;
            obj_th->class_name_ = class_name;
            objects_[index] = obj_th;
            return obj_th;
        }
        else if (typeidx == TYPE_TABLE)
        {
            int size = read_int(is);

            std::shared_ptr<cpptorch::object_table> obj_tbl = std::make_shared<cpptorch::object_table>();
            for (int i = 0; i < size; i++)
            {
                std::shared_ptr<cpptorch::object> k = read_object(is);
                if (k->type_ == cpptorch::object_type_string)
                {
                    std::shared_ptr<cpptorch::object_string> ks = std::static_pointer_cast<cpptorch::object_string>(k);
                    obj_tbl->kv_[ks->str_] = read_object(is);
                }
                else if (k->type_ == cpptorch::object_type_number)
                {
                    std::shared_ptr<cpptorch::object_number> kn = std::static_pointer_cast<cpptorch::object_number>(k);
                    size_t idx = (size_t)round(kn->num_) - 1;
                    obj_tbl->array_.resize(idx + 1);
                    obj_tbl->array_[idx] = read_object(is);
                }
                else
                {
                    assert(0);
                }
            }
            objects_[index] = obj_tbl;
            return obj_tbl;
        }
        break;
    }

    assert(0);
    return nullptr;
}


//////////////////////////////////////////////////////////////////////////


std::shared_ptr<cpptorch::object> cpptorch::load(std::istream &is)
{
    object_loader me;
    return me.read_object(is);
}
