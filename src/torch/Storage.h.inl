#pragma once
#include "../../include/torch/Storage.h"
#include "../th_wrapper.h"

#include <sstream>
#include <assert.h>
#include <string.h>


template<typename T, GPUFlag F>
cpptorch::Storage<T, F>::Storage(typename THTrait<T, F>::Storage *th) : th_(th)
{
    if (th_)
    {
        cpptorch::th::Storage<T, F>::retain(th_);
    }
}

template<typename T, GPUFlag F>
cpptorch::Storage<T, F>::~Storage()
{
    if (th_)
    {
        cpptorch::th::Storage<T, F>::release(th_);
        th_ = nullptr;
    }
}

template<typename T, GPUFlag F>
cpptorch::Storage<T, F>& cpptorch::Storage<T, F>::operator =(const cpptorch::Storage<T, F> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Storage<T, F>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Storage<T, F>::retain(th_);
        }
    }
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Storage<T, F>& cpptorch::Storage<T, F>::operator =(Storage<T, F> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Storage<T, F>::release(th_);
        th_ = nullptr;
    }
    if (other.th_)
    {
        th_ = other.th_;
        other.th_ = nullptr;
    }
    return *this;
}

template<typename T, GPUFlag F>
int cpptorch::Storage<T, F>::size() const
{
    return th_ ? cpptorch::th::Storage<T, F>::size(th_) : 0;
}

template<typename T, GPUFlag F>
const T* cpptorch::Storage<T, F>::data() const
{
    return th_ ? cpptorch::th::Storage<T, F>::data(th_) : nullptr;
}

template<typename T, GPUFlag F>
T* cpptorch::Storage<T, F>::data()
{
    return th_ ? cpptorch::th::Storage<T, F>::data(th_) : nullptr;
}

//////////////////////////////////////////////////////////////////////////

template<typename T, GPUFlag F>
void cpptorch::Storage<T, F>::create()
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Storage<T, F>::newWithData(nullptr, 0, false);
}

template<typename T, GPUFlag F>
void cpptorch::Storage<T, F>::unserialze(const T *ptr_src, long count, bool take_ownership_of_data)
{
    th_ = cpptorch::th::Storage<T, F>::newWithData(ptr_src, count, take_ownership_of_data);
}
