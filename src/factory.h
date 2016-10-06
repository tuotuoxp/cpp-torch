#pragma once
#include <string>
#include <memory>
#include <map>


#define MAP_CLASS(T)        (new class_factory_impl<T>())


class class_factory_base
{
public:
    virtual void* operator()() const = 0;
};

template <class T>
class class_factory_impl : public class_factory_base
{
public:
    void* operator()() const override { return new T(); }
};


template<class Base>
class class_factory
{
public:
    // add_map("MyClass", MAP_CLASS(MyClass));
    void add_map(const std::string &class_name, class_factory_base *instance)
    {
        factory_.insert(std::make_pair(class_name, std::shared_ptr<class_factory_base>(instance)));
    }

    std::shared_ptr<Base> create_from_factory(const std::string &class_name) const
    {
        auto factory = factory_.find(class_name);
        std::shared_ptr<Base> obj;
        if (factory != factory_.end())
        {
            return std::shared_ptr<Base>(reinterpret_cast<Base*>((*factory->second)()));
        }
        return nullptr;
    }


private:

    // factory mapper: class name -> factory instance
    std::map<std::string, std::shared_ptr<class_factory_base>> factory_;
};
