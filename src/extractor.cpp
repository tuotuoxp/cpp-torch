#include "extractor.h"

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
long* model_extractor::read_array<long>(std::istream &is, size_t count)
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

void object_torch::read(model_extractor &extractor, std::istream &is)
{
    data_ = extractor.read_object(is);
}

void object_torch_tensor::read(model_extractor &extractor, std::istream &is)
{
    dimension_ = model_extractor::read_int(is);
    if (dimension_ > 0)
    {
        size_ = model_extractor::read_array<long>(is, dimension_);
        stride_ = model_extractor::read_array<long>(is, dimension_);
    }
    storage_offset_ = model_extractor::read_long(is) - 1;
    object_torch::read(extractor, is);
}

//////////////////////////////////////////////////////////////////////////

model_extractor::model_extractor()
{
    factory_.add_map("torch.FloatTensor", MAP_CLASS(object_torch_tensor));
    factory_.add_map("torch.LongTensor", MAP_CLASS(object_torch_tensor));
    factory_.add_map("torch.FloatStorage", MAP_CLASS(object_torch_storage<float>));
    factory_.add_map("torch.LongStorage", MAP_CLASS(object_torch_storage<long>));
}

std::shared_ptr<object> model_extractor::read_object(std::istream &is)
{
    int typeidx = read_int(is);

    switch (typeidx)
    {
    case TYPE_NIL:
        return std::make_shared<object>();
    case TYPE_NUMBER:
    {
        std::shared_ptr<object_number> on = std::make_shared<object_number>();
        on->num_ = read_double(is);
        return on;
    }
    case TYPE_STRING:
    {
        std::shared_ptr<object_string> os = std::make_shared<object_string>();
        os->str_ = read_string_with_length(is);
        return os;
    }
    case TYPE_BOOLEAN:
    {
        std::shared_ptr<object_boolean> ob = std::make_shared<object_boolean>();
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

            std::shared_ptr<object_torch> obj_th = factory_.create_from_factory(class_name);
            if (obj_th == nullptr)
            {
                obj_th = std::make_shared<object_torch>();
            }
            obj_th->version_ = version_number;
            obj_th->index_ = index;
            obj_th->class_name_ = class_name;
            obj_th->read(*this, is);
            objects_[index] = obj_th;
            return obj_th;
        }
        else if (typeidx == TYPE_TABLE)
        {
            int size = read_int(is);

            std::shared_ptr<object_table> obj_tbl = std::make_shared<object_table>();
            for (int i = 0; i < size; i++)
            {
                std::shared_ptr<object> k = read_object(is);
                if (k->type_ == object_type_string)
                {
                    std::shared_ptr<object_string> ks = std::static_pointer_cast<object_string>(k);
                    obj_tbl->kv_[ks->str_] = read_object(is);
                }
                else if (k->type_ == object_type_number)
                {
                    std::shared_ptr<object_number> kn = std::static_pointer_cast<object_number>(k);
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
