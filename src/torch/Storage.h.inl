#pragma once
#include "../../include/torch/Storage.h"
#include "../../include/th_wrapper.h"

#include <sstream>
#include <assert.h>
#include <TH/TH.h>


static void *mallocWrapper(void* ctx, long size) { return malloc(size); }
static void *reallocWrapper(void* ctx, void* ptr, long size) { return realloc(ptr, size); }
static void freeWrapper(void* ctx, void* ptr) { free(ptr); }


template<class TStorage>
cpptorch::Storage<TStorage>::Storage(typename TStorage::TH *th) : th_(th)
{
    if (th_)
    {
        cpptorch::th::Storage<TStorage>::retain(th_);
    }
}

template<class TStorage>
cpptorch::Storage<TStorage>::~Storage()
{
    if (th_)
    {
        cpptorch::th::Storage<TStorage>::release(th_);
        th_ = nullptr;
    }
}

template<class TStorage>
cpptorch::Storage<TStorage>& cpptorch::Storage<TStorage>::operator =(const cpptorch::Storage<TStorage> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Storage<TStorage>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Storage<TStorage>::retain(th_);
        }
    }
    return *this;
}

template<class TStorage>
cpptorch::Storage<TStorage>& cpptorch::Storage<TStorage>::operator =(Storage<TStorage> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Storage<TStorage>::release(th_);
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
int cpptorch::Storage<TStorage>::size() const
{
    return th_ ? cpptorch::th::Storage<TStorage>::size(th_) : 0;
}

template<class TStorage>
const typename TStorage::Base* cpptorch::Storage<TStorage>::data() const
{
    return th_ ? cpptorch::th::Storage<TStorage>::data(th_) : nullptr;
}

template<class TStorage>
typename TStorage::Base* cpptorch::Storage<TStorage>::data()
{
    return th_ ? cpptorch::th::Storage<TStorage>::data(th_) : nullptr;
}

//////////////////////////////////////////////////////////////////////////

template<class TStorage>
void cpptorch::Storage<TStorage>::unserialze(const typename TStorage::Base *ptr_src, long size, bool take_ownership_of_data)
{
    if (!take_ownership_of_data)
    {
        long sz = size * sizeof(typename TStorage::Base);
        typename TStorage::Base *ptr = (typename TStorage::Base*)malloc(sz);
        memcpy(ptr, ptr_src, sz);
        ptr_src = ptr;
    }
    static THAllocator allocater_ =
    {
        mallocWrapper,
        reallocWrapper,
        freeWrapper
    };
    th_ = cpptorch::th::Storage<TStorage>::newWithDataAndAllocator(const_cast<typename TStorage::Base*>(ptr_src), size, &allocater_, nullptr);
}
