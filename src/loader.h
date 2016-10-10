#pragma once
#include "factory.h"
#include "../include/builder.h"

#include <cmath>


class object_loader
{
public:

    object_loader();

    std::shared_ptr<cpptorch::object> read_object(std::istream &is);



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
    std::map<int, std::shared_ptr<cpptorch::object>> objects_;


    class class_factory_base
    {
    public:
        virtual cpptorch::object_torch* create(object_loader &extractor, std::istream &is) const = 0;
    };

    template <class T>
    class class_factory_impl : public class_factory_base
    {
    public:
        virtual cpptorch::object_torch* create(object_loader &extractor, std::istream &is) const
        {
            T *t = new T();
            t->read(extractor, is);
            return static_cast<cpptorch::object_torch*>(t);
        }
    };
    // class name to torch object
    std::map<std::string, std::shared_ptr<class_factory_base>> factory_;
};

