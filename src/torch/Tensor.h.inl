#pragma once
#include "../../include/torch/Tensor.h"
#include "TensorPrint.h.inl"
#include "../util.h"


void specifyFully(std::vector<long> &sto_size, int nElements);


template<typename T>
cpptorch::Tensor<T>::Tensor(bool auto_create) : th_(nullptr)
{
    if (auto_create)
    {
        create();
    }
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator =(const cpptorch::Tensor<T> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Tensor<T>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Tensor<T>::retain(th_);
        }
    }
    return *this;
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator =(Tensor<T> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Tensor<T>::release(th_);
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
cpptorch::Tensor<T>::~Tensor()
{
    if (th_)
    {
        cpptorch::th::Tensor<T>::release(th_);
        th_ = nullptr;
    }
}

template<typename T>
const std::string cpptorch::Tensor<T>::name() const
{
    if (std::is_same<T, long>::value)
    {
        return "torch.LongTensor";
    }
    else if (std::is_same<T, float>::value)
    {
        return "torch.FloatTensor";
    }
    return "torch.Tensor";
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
void cpptorch::Tensor<T>::create()
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Tensor<T>::create();
}

template<typename T>
void cpptorch::Tensor<T>::create(const Storage<T> &storage, long storage_offset,
    const Storage<long> &size)
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Tensor<T>::newWithStorage(storage, storage_offset, size, nullptr);
}

template<typename T>
void cpptorch::Tensor<T>::create(const Storage<T> &storage, long storage_offset,
    const Storage<long> &size, const Storage<long> &stride)
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Tensor<T>::newWithStorage(storage, storage_offset, size, stride);
}

template<typename T>
void cpptorch::Tensor<T>::resize(const Storage<long> &size)
{
    cpptorch::th::Tensor<T>::resize(th_, size, nullptr);
}

template<typename T>
void cpptorch::Tensor<T>::resize(const Storage<long> &size, const Storage<long> &stride)
{
    cpptorch::th::Tensor<T>::resize(th_, size, stride);
}

template<typename T>
void cpptorch::Tensor<T>::resizeAs(const Tensor<T> &src)
{
    cpptorch::th::Tensor<T>::resizeAs(th_, src.th_);
}

template<typename T>
void cpptorch::Tensor<T>::copy(const Tensor<T> &src)
{
    cpptorch::th::Tensor<T>::copy(th_, src.th_);
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
cpptorch::Storage<T> cpptorch::Tensor<T>::storage() const
{
    return cpptorch::Storage<T>(th_ ? cpptorch::th::Tensor<T>::storage(th_) : nullptr);
}

template<typename T>
long cpptorch::Tensor<T>::storageOffset() const
{
    return th_ ? cpptorch::th::Tensor<T>::storageOffset(th_) : 0;
}

template<typename T>
std::vector<long> cpptorch::Tensor<T>::size() const
{
    std::vector<long> v;
    if (th_)
    {
        THTrait<long>::Storage *th = cpptorch::th::Tensor<T>::size(th_);
        long *p = cpptorch::th::Storage<long>::data(th);
        v.assign(p, p + dim());
        cpptorch::th::Storage<long>::release(th);
    }
    return v;
}

template<typename T>
long cpptorch::Tensor<T>::size(int dim) const
{
    return cpptorch::th::Tensor<T>::size(th_, dim);
}

template<typename T>
std::vector<long> cpptorch::Tensor<T>::stride() const
{
    std::vector<long> v;
    if (th_)
    {
        THTrait<long>::Storage *th = cpptorch::th::Tensor<T>::stride(th_);
        long *p = cpptorch::th::Storage<long>::data(th);
        v.assign(p, p + dim());
        cpptorch::th::Storage<long>::release(th);
    }
    return v;
}

template<typename T>
int cpptorch::Tensor<T>::dim() const
{
    return cpptorch::th::Tensor<T>::nDimension(th_);
}

template<typename T>
T* cpptorch::Tensor<T>::data() const
{
    return cpptorch::th::Tensor<T>::data(th_);
}

template<typename T>
cpptorch::Tensor<T>::operator T() const
{
    asserter(nElement() == 1) << "only an 1-D tensor with ONE element can be cast to number";
    return data()[0];
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
bool cpptorch::Tensor<T>::isContiguous() const
{
    return cpptorch::th::Tensor<T>::isContiguous(th_) != 0;
}

template<typename T>
int cpptorch::Tensor<T>::nElement() const
{
    return cpptorch::th::Tensor<T>::nElement(th_);
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::narrow(int dimension, long firstIndex, long size) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::narrow(out, th_, dimension, firstIndex, size);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::select(int dimension, long sliceIndex) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::select(out, th_, dimension, sliceIndex);
    return out;
}

template<typename T>
template<class TIterator>
cpptorch::Tensor<T> cpptorch::Tensor<T>::view(const TIterator begin, const TIterator end) const
{
    std::vector<long> sz(begin, end);
    int origNElement = nElement();
    specifyFully(sz, origNElement);

    std::vector<long> ss = size();
    assert(isContiguous() && "expecting a contiguous tensor");
    cpptorch::Tensor<T> view;
    view.create(storage(), storageOffset(), sz);
    assert(view.nElement() == origNElement && "Wrong size for view. ");
    return view;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::expand(const std::vector<long> &sz) const
{
    int tensor_dim = dim();
    std::vector<long> tensor_stride = stride();
    std::vector<long> tensor_size = size();

    assert(sz.size() == tensor_dim && "the number of dimensions provided must equal tensor.dim()");

    // create a new geometry for tensor:
    for (int i = 0; i < tensor_dim; i++)
    {
        if (tensor_size[i] == 1)
        {
            tensor_size[i] = sz[i];
            tensor_stride[i] = 0;
        }
        else
        {
            assert(tensor_size[i] == sz[i] && "incorrect size: only supporting singleton expansion (size=1)");
        }
    }

    // create new view, with singleton expansion:
    cpptorch::Tensor<T> output;
    output.create(storage(), storageOffset(), tensor_size, tensor_stride);
    return output;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::t() const
{
    cpptorch::Tensor<T> output(true);
    cpptorch::th::Tensor<T>::transpose(output, th_, 0, 1);
    return output;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::operator [] (const std::initializer_list<long> &inputs) const
{
    std::vector<long> tsize = size();
    std::vector<long> tstride = stride();
    asserter(inputs.size() <= tsize.size()) << "invalid size";

    long index = storageOffset();
    int dim = 0;
    for (auto it : inputs)
    {
        long z = it;
        if (z < 0)
        {
            z = tsize[dim] + z + 1;
        }
        asserter(z >= 0 && z < tsize[dim]) << "index out of bound";
        index += z * tstride[dim];
        dim++;
    }
    tsize.erase(tsize.begin(), tsize.begin() + dim);
    tstride.erase(tstride.begin(), tstride.begin() + dim);

    cpptorch::Tensor<T> output;
    if (tsize.size() == 0)
    {
        tsize.push_back(1);
        tstride.push_back(1);
    }
    output.create(storage(), index, cpptorch::Storage<long>(tsize), cpptorch::Storage<long>(tstride));
    return output;
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
void cpptorch::Tensor<T>::fill(T val)
{
    cpptorch::th::Tensor<T>::fill(th_, val);
}

template<typename T>
void cpptorch::Tensor<T>::abs()
{
    cpptorch::th::Tensor<T>::abs(th_, th_);
}

template<typename T>
void cpptorch::Tensor<T>::addmv(T beta, const Tensor<T> &t,
    T alpha, const Tensor<T> &matrix, const Tensor<T> &vector)
{
    cpptorch::th::Tensor<T>::addmv(th_, beta, t, alpha, matrix, vector);
}

template<typename T>
void cpptorch::Tensor<T>::addmm(T beta, const Tensor<T> &t,
    T alpha, const Tensor<T> &matrix1, const Tensor<T> &matrix2)
{
    cpptorch::th::Tensor<T>::addmm(th_, beta, t, alpha, matrix1, matrix2);
}

template<typename T>
void cpptorch::Tensor<T>::addr(T beta, const Tensor<T> &t,
    T alpha, const Tensor<T> &vector1, const Tensor<T> &vector2)
{
    cpptorch::th::Tensor<T>::addr(th_, beta, t, alpha, vector1, vector2);
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator += (T val)
{
    cpptorch::th::Tensor<T>::add(th_, th_, val);
    return *this;
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator += (const cpptorch::Tensor<T> &other)
{
    cpptorch::th::Tensor<T>::cadd(th_, th_, 1, other);
    return *this;
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator -= (const cpptorch::Tensor<T> &other)
{
    cpptorch::th::Tensor<T>::cadd(th_, th_, -1, other);
    return *this;
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator *= (T val)
{
    cpptorch::th::Tensor<T>::mul(th_, th_, val);
    return *this;
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator *= (const cpptorch::Tensor<T> &other)
{
    cpptorch::th::Tensor<T>::cmul(th_, th_, other);
    return *this;
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator ^= (T val)
{
    cpptorch::th::Tensor<T>::pow(th_, th_, val);
    return *this;
}

template<typename T>
cpptorch::Tensor<T>& cpptorch::Tensor<T>::operator ^= (const Tensor<T> &other)
{
    cpptorch::th::Tensor<T>::cpow(th_, th_, other);
    return *this;
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
T cpptorch::Tensor<T>::minall() const
{
    return th_ ? cpptorch::th::Tensor<T>::minall(th_) : 0;
}

template<typename T>
T cpptorch::Tensor<T>::maxall() const
{
    return th_ ? cpptorch::th::Tensor<T>::maxall(th_) : 0;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::max(int dimension) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::max(out.th_, th_, dimension);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::sum(int dimension) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::sum(out.th_, th_, dimension);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::operator +(T val) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::add(out.th_, th_, val);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::operator +(const Tensor<T> &other) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::cadd(out, th_, 1, other);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::operator -(T val) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::add(out, th_, -val);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::operator -(const Tensor<T> &other) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::cadd(out, th_, (T)-1.0, other);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::operator /(const Tensor<T> &other) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::cdiv(out, th_, other);
    return out;
}

template<typename T>
cpptorch::Tensor<T> cpptorch::Tensor<T>::operator ^(T val) const
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::pow(out, th_, val);
    return out;
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


/*
 One of the size elements can be -1, a new LongStorage is then returned.
 The length of the unspecified dimension is inferred from the number of remaining elements.
*/
void specifyFully(std::vector<long> &sz, int nElements)
{
    int nCoveredElements = 1;
    int remainingDim = -1;
    for (int i = 0; i < (int)sz.size(); i++)
    {
        long wantedDimSize = sz[i];
        if (wantedDimSize == -1)
        {
            assert(remainingDim == -1 && "Only one of torch.view dimensions can be -1.");
            remainingDim = i;
        }
        else
        {
            nCoveredElements = nCoveredElements * wantedDimSize;
        }
    }

    if (remainingDim >= 0)
    {
        assert(nElements % nCoveredElements == 0 && "The number of covered elements is not a multiple of all elements.");
        sz[remainingDim] = nElements / nCoveredElements;
    }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename T>
cpptorch::Tensor<T> cpptorch::abs(const cpptorch::Tensor<T> &t)
{
    cpptorch::Tensor<T> out(true);
    cpptorch::th::Tensor<T>::abs(out, t);
    return out;
}
