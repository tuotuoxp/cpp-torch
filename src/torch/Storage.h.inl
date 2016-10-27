#pragma once
#include "../../include/torch/Storage.h"
#include "../th_wrapper.h"

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
    th_ = cpptorch::th::Storage<T,C>::newWithData(nullptr, 0, false);
}

template<typename T, bool C>
void cpptorch::Storage<T,C>::unserialze(const T *ptr_src, long count, bool take_ownership_of_data)
{
    th_ = cpptorch::th::Storage<T, C>::newWithData(ptr_src, sizeof(T) * count, take_ownership_of_data);
}
