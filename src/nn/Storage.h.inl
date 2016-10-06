#pragma once
#include "Storage.h"
#include "../extractor.h"
#include "../builder.h"
#include "../th_wrapper.h"

#include <sstream>


template<class TStorage>
THAllocator nn::Storage<TStorage>::allocater_ =
{
    nn::Storage<TStorage>::mallocWrapper,
    nn::Storage<TStorage>::reallocWrapper,
    nn::Storage<TStorage>::freeWrapper
};


template<class TStorage>
nn::Storage<TStorage>::Storage(typename TStorage::THStorage *th) : th_(th)
{
    if (th_)
    {
        THWrapper::Storage<TStorage>::retain(th_);
    }
}

template<class TStorage>
nn::Storage<TStorage>::~Storage()
{
    if (th_)
    {
        THWrapper::Storage<TStorage>::release(th_);
        th_ = nullptr;
    }
}

template<class TStorage>
nn::Storage<TStorage>& nn::Storage<TStorage>::operator =(const nn::Storage<TStorage> &other)
{
    if (this != &other) {
        if (th_)
        {
            THWrapper::Storage<TStorage>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            THWrapper::Storage<TStorage>::retain(th_);
        }
    }
    return *this;
}

template<class TStorage>
nn::Storage<TStorage>& nn::Storage<TStorage>::operator =(Storage<TStorage> &&other)
{
    assert(this != &other);
    if (th_)
    {
        THWrapper::Storage<TStorage>::release(th_);
        th_ = nullptr;
    }
    if (other.th_)
    {
        th_ = other.th_;
        other.th_ = nullptr;
    }
    return *this;
}

template<class TStorage>
int nn::Storage<TStorage>::size() const
{
    return th_ ? THWrapper::Storage<TStorage>::size(th_) : 0;
}

template<class TStorage>
const typename TStorage::StorageBase* nn::Storage<TStorage>::data() const
{
    return th_ ? THWrapper::Storage<TStorage>::data(th_) : nullptr;
}

template<class TStorage>
typename TStorage::StorageBase* nn::Storage<TStorage>::data()
{
    return th_ ? THWrapper::Storage<TStorage>::data(th_) : nullptr;
}

//////////////////////////////////////////////////////////////////////////

template<class TStorage>
void nn::Storage<TStorage>::unserialze(const typename TStorage::StorageBase *ptr_src, long size, bool take_ownership_of_data)
{
    if (!take_ownership_of_data)
    {
        long sz = size * sizeof(typename TStorage::StorageBase);
        typename TStorage::StorageBase *ptr = (typename TStorage::StorageBase*)malloc(sz);
        memcpy(ptr, ptr_src, sz);
        ptr_src = ptr;
    }
    th_ = THWrapper::Storage<TStorage>::newWithDataAndAllocator(const_cast<typename TStorage::StorageBase*>(ptr_src), size, &allocater_, nullptr);
}


template<class TStorage> template <class TIterator>
void nn::Storage<TStorage>::unserialze(const TIterator begin, const TIterator end)
{
    long size = (long)(end - begin);
    size_t sz = size * sizeof(typename TStorage::StorageBase);
    typename TStorage::StorageBase *ptr = (typename TStorage::StorageBase*)malloc(sz);
    int i = 0;
    for (auto it = begin; it != end; it++, i++)
    {
        ptr[i] = *it;
    }
    assert(th_ == nullptr);
    th_ = THWrapper::Storage<TStorage>::newWithDataAndAllocator(ptr, size, &allocater_, nullptr);
}
