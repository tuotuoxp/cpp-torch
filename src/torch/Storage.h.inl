#pragma once
#include "../../include/torch/Storage.h"
#include "../th_wrapper.h"
#include "../allocator.h"

#include <sstream>
#include <assert.h>
#include <string.h>


template<typename T, bool C>
cpptorch::Storage<T,C>::Storage(typename THTrait<T,C>::Storage *th) : th_(th)
{
    if (th_)
    {
        cpptorch::th::Storage<T,C>::retain(th_);
    }
}

template<typename T, bool C>
cpptorch::Storage<T,C>::~Storage()
{
    if (th_)
    {
        cpptorch::th::Storage<T,C>::release(th_);
        th_ = nullptr;
    }
}

template<typename T, bool C>
cpptorch::Storage<T,C>& cpptorch::Storage<T,C>::operator =(const cpptorch::Storage<T,C> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Storage<T,C>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Storage<T,C>::retain(th_);
        }
    }
    return *this;
}

template<typename T, bool C>
cpptorch::Storage<T,C>& cpptorch::Storage<T,C>::operator =(Storage<T,C> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Storage<T,C>::release(th_);
        th_ = nullptr;
    }
    if (other.th_)
    {
        th_ = other.th_;
        other.th_ = nullptr;
    }
    return *this;
}

template<typename T, bool C>
int cpptorch::Storage<T,C>::size() const
{
    return th_ ? cpptorch::th::Storage<T,C>::size(th_) : 0;
}

template<typename T, bool C>
const T* cpptorch::Storage<T,C>::data() const
{
    return th_ ? cpptorch::th::Storage<T,C>::data(th_) : nullptr;
}

template<typename T, bool C>
T* cpptorch::Storage<T,C>::data()
{
    return th_ ? cpptorch::th::Storage<T,C>::data(th_) : nullptr;
}

//////////////////////////////////////////////////////////////////////////

template<typename T, bool C>
void cpptorch::Storage<T,C>::create()
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Storage<T,C>::newWithAllocator(cpptorch::allocator::get(), cpptorch::allocator::requestIndex(0));
}

template<typename T, bool C>
void cpptorch::Storage<T,C>::unserialze(const T *ptr_src, long size, bool take_ownership_of_data)
{
    if (!take_ownership_of_data)
    {
        long sz = size * sizeof(T);
        T *ptr = (T*)malloc(sz);
        memcpy(ptr, ptr_src, sz);
        ptr_src = ptr;
    }
    th_ = cpptorch::th::Storage<T,C>::newWithDataAndAllocator(const_cast<T*>(ptr_src), size,
        cpptorch::allocator::get(), cpptorch::allocator::requestIndex(size * sizeof(T)));
}
