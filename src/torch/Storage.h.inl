#pragma once
#include "../../include/torch/Storage.h"
#include "../th_wrapper.h"
#include "../allocator.h"

#include <sstream>
#include <assert.h>
#include <string.h>


template<typename T>
cpptorch::Storage<T>::Storage(typename THTrait<T>::Storage *th) : th_(th)
{
    if (th_)
    {
        cpptorch::th::Storage<T>::retain(th_);
    }
}

template<typename T>
cpptorch::Storage<T>::~Storage()
{
    if (th_)
    {
        cpptorch::th::Storage<T>::release(th_);
        th_ = nullptr;
    }
}

template<typename T>
cpptorch::Storage<T>& cpptorch::Storage<T>::operator =(const cpptorch::Storage<T> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Storage<T>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Storage<T>::retain(th_);
        }
    }
    return *this;
}

template<typename T>
cpptorch::Storage<T>& cpptorch::Storage<T>::operator =(Storage<T> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Storage<T>::release(th_);
        th_ = nullptr;
    }
    if (other.th_)
    {
        th_ = other.th_;
        other.th_ = nullptr;
    }
    return *this;
}

template<typename T>
int cpptorch::Storage<T>::size() const
{
    return th_ ? cpptorch::th::Storage<T>::size(th_) : 0;
}

template<typename T>
const T* cpptorch::Storage<T>::data() const
{
    return th_ ? cpptorch::th::Storage<T>::data(th_) : nullptr;
}

template<typename T>
T* cpptorch::Storage<T>::data()
{
    return th_ ? cpptorch::th::Storage<T>::data(th_) : nullptr;
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
void cpptorch::Storage<T>::create()
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Storage<T>::newWithAllocator(cpptorch::allocator::get(), cpptorch::allocator::requestIndex());
}

template<typename T>
void cpptorch::Storage<T>::unserialze(const T *ptr_src, long size, bool take_ownership_of_data)
{
    if (!take_ownership_of_data)
    {
        long sz = size * sizeof(T);
        T *ptr = (T*)malloc(sz);
        memcpy(ptr, ptr_src, sz);
        ptr_src = ptr;
    }
    th_ = cpptorch::th::Storage<T>::newWithDataAndAllocator(const_cast<T*>(ptr_src), size,
        cpptorch::allocator::get(), cpptorch::allocator::requestIndex());
}
