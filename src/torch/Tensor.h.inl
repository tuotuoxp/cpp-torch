#pragma once
#include "../../include/torch/Tensor.h"
#include "TensorPrint.h.inl"
#include "../util.h"


void specifyFully(std::vector<long> &sto_size, int nElements);


template<typename T, bool C>
cpptorch::Tensor<T,C>::Tensor(bool auto_create) : th_(nullptr)
{
    if (auto_create)
    {
        create();
    }
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator =(const cpptorch::Tensor<T,C> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Tensor<T,C>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Tensor<T,C>::retain(th_);
        }
    }
    return *this;
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator =(Tensor<T,C> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Tensor<T,C>::release(th_);
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
cpptorch::Tensor<T,C>::~Tensor()
{
    if (th_)
    {
        cpptorch::th::Tensor<T,C>::release(th_);
        th_ = nullptr;
    }
}

template<typename T, bool C>
const std::string cpptorch::Tensor<T,C>::name() const
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

template<typename T, bool C>
void cpptorch::Tensor<T,C>::create()
{
    assert(th_ == nullptr);
    Storage<T,C> s;
    s.create();
    th_ = cpptorch::th::Tensor<T,C>::newWithStorage(s, 0, 0, nullptr, nullptr);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::create(const Storage<T,C> &storage, long storage_offset, int dim,
    const long *size, const long *stride)
{
    assert(th_ == nullptr);
    if (!stride)
    {
        long *new_stride = (long*)alloca(sizeof(long) * dim);
        for (int i = 0; i < dim; i++)
        {
            new_stride[i] = -1;
        }
        stride = new_stride;
    }
    th_ = cpptorch::th::Tensor<T,C>::newWithStorage(storage, storage_offset, dim, size, stride);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::resize(const Storage<long, false> &size)
{
    cpptorch::th::Tensor<T,C>::resize(th_, size, nullptr);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::resize(const Storage<long, false> &size, const Storage<long, false> &stride)
{
    cpptorch::th::Tensor<T,C>::resize(th_, size, stride);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::resizeAs(const Tensor<T,C> &src)
{
    cpptorch::th::Tensor<T,C>::resizeAs(th_, src.th_);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::copy(const Tensor<T,C> &src)
{
    cpptorch::th::Tensor<T,C>::copy(th_, src.th_);
}

//////////////////////////////////////////////////////////////////////////

template<typename T, bool C>
cpptorch::Storage<T,C> cpptorch::Tensor<T,C>::storage() const
{
    return cpptorch::Storage<T,C>(th_ ? cpptorch::th::Tensor<T,C>::storage(th_) : nullptr);
}

template<typename T, bool C>
long cpptorch::Tensor<T,C>::storageOffset() const
{
    return th_ ? cpptorch::th::Tensor<T,C>::storageOffset(th_) : 0;
}

template<typename T, bool C>
std::vector<long> cpptorch::Tensor<T,C>::size() const
{
    std::vector<long> v;
    if (th_)
    {
        Storage<long, false> sz(cpptorch::th::Tensor<T, C>::size(th_));
        long *p = sz.data();
        v.assign(p, p + dim());
    }
    return v;
}

template<typename T, bool C>
long cpptorch::Tensor<T,C>::size(int dim) const
{
    return cpptorch::th::Tensor<T,C>::size(th_, dim);
}

template<typename T, bool C>
std::vector<long> cpptorch::Tensor<T,C>::stride() const
{
    std::vector<long> v;
    if (th_)
    {
        THTrait<long, false>::Storage *th = cpptorch::th::Tensor<T,C>::stride(th_);
        long *p = cpptorch::th::Storage<long, false>::data(th);
        v.assign(p, p + dim());
        cpptorch::th::Storage<long, false>::release(th);
    }
    return v;
}

template<typename T, bool C>
int cpptorch::Tensor<T,C>::dim() const
{
    return cpptorch::th::Tensor<T,C>::nDimension(th_);
}

template<typename T, bool C>
T* cpptorch::Tensor<T,C>::data() const
{
    return cpptorch::th::Tensor<T,C>::data(th_);
}

template<typename T, bool C>
cpptorch::Tensor<T,C>::operator T() const
{
    asserter(nElement() == 1) << "only an 1-D tensor with ONE element can be cast to number";
    return data()[0];
}

//////////////////////////////////////////////////////////////////////////

template<typename T, bool C>
bool cpptorch::Tensor<T,C>::isContiguous() const
{
    return cpptorch::th::Tensor<T,C>::isContiguous(th_) != 0;
}

template<typename T, bool C>
int cpptorch::Tensor<T,C>::nElement() const
{
    return cpptorch::th::Tensor<T,C>::nElement(th_);
}

//////////////////////////////////////////////////////////////////////////

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::narrow(int dimension, long firstIndex, long size) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::narrow(out, th_, dimension, firstIndex, size);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::select(int dimension, long sliceIndex) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::select(out, th_, dimension, sliceIndex);
    return out;
}

template<typename T, bool C>
template<class TIterator>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::view(const TIterator begin, const TIterator end) const
{
    std::vector<long> sz(begin, end);
    int origNElement = nElement();
    specifyFully(sz, origNElement);

    std::vector<long> ss = size();
    assert(isContiguous() && "expecting a contiguous tensor");
    cpptorch::Tensor<T,C> view;
    view.create(storage(), storageOffset(), (int)sz.size(), &sz[0]);
    assert(view.nElement() == origNElement && "Wrong size for view. ");
    return view;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::expand(const std::vector<long> &sz) const
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
    cpptorch::Tensor<T,C> output;
    output.create(storage(), storageOffset(), tensor_dim, &tensor_size[0], &tensor_stride[0]);
    return output;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::t() const
{
    cpptorch::Tensor<T,C> output(true);
    cpptorch::th::Tensor<T,C>::transpose(output, th_, 0, 1);
    return output;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::operator [] (const std::initializer_list<long> &inputs) const
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

    cpptorch::Tensor<T,C> output;
    if (tsize.size() == 0)
    {
        tsize.push_back(1);
        tstride.push_back(1);
    }
    output.create(storage(), index, (int)tsize.size(), &tsize[0], &tstride[0]);
    return output;
}

//////////////////////////////////////////////////////////////////////////

template<typename T, bool C>
void cpptorch::Tensor<T,C>::fill(T val)
{
    cpptorch::th::Tensor<T,C>::fill(th_, val);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::abs()
{
    cpptorch::th::Tensor<T,C>::abs(th_, th_);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::addmv(T beta, const Tensor<T,C> &t,
    T alpha, const Tensor<T,C> &matrix, const Tensor<T,C> &vector)
{
    cpptorch::th::Tensor<T,C>::addmv(th_, beta, t, alpha, matrix, vector);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::addmm(T beta, const Tensor<T,C> &t,
    T alpha, const Tensor<T,C> &matrix1, const Tensor<T,C> &matrix2)
{
    cpptorch::th::Tensor<T,C>::addmm(th_, beta, t, alpha, matrix1, matrix2);
}

template<typename T, bool C>
void cpptorch::Tensor<T,C>::addr(T beta, const Tensor<T,C> &t,
    T alpha, const Tensor<T,C> &vector1, const Tensor<T,C> &vector2)
{
    cpptorch::th::Tensor<T,C>::addr(th_, beta, t, alpha, vector1, vector2);
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator += (T val)
{
    cpptorch::th::Tensor<T,C>::add(th_, th_, val);
    return *this;
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator += (const cpptorch::Tensor<T,C> &other)
{
    cpptorch::th::Tensor<T,C>::cadd(th_, th_, 1, other);
    return *this;
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator -= (const cpptorch::Tensor<T,C> &other)
{
    cpptorch::th::Tensor<T,C>::cadd(th_, th_, -1, other);
    return *this;
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator *= (T val)
{
    cpptorch::th::Tensor<T,C>::mul(th_, th_, val);
    return *this;
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator *= (const cpptorch::Tensor<T,C> &other)
{
    cpptorch::th::Tensor<T,C>::cmul(th_, th_, other);
    return *this;
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator ^= (T val)
{
    cpptorch::th::Tensor<T,C>::pow(th_, th_, val);
    return *this;
}

template<typename T, bool C>
cpptorch::Tensor<T,C>& cpptorch::Tensor<T,C>::operator ^= (const Tensor<T,C> &other)
{
    cpptorch::th::Tensor<T,C>::cpow(th_, th_, other);
    return *this;
}

//////////////////////////////////////////////////////////////////////////

template<typename T, bool C>
T cpptorch::Tensor<T,C>::minall() const
{
    return th_ ? cpptorch::th::Tensor<T,C>::minall(th_) : 0;
}

template<typename T, bool C>
T cpptorch::Tensor<T,C>::maxall() const
{
    return th_ ? cpptorch::th::Tensor<T,C>::maxall(th_) : 0;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::max(int dimension) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::max(out.th_, th_, dimension);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::sum(int dimension) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::sum(out.th_, th_, dimension);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::operator +(T val) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::add(out.th_, th_, val);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::operator +(const Tensor<T,C> &other) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::cadd(out, th_, 1, other);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::operator -(T val) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::add(out, th_, -val);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::operator -(const Tensor<T,C> &other) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::cadd(out, th_, (T)-1.0, other);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::operator /(const Tensor<T,C> &other) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::cdiv(out, th_, other);
    return out;
}

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::Tensor<T,C>::operator ^(T val) const
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::pow(out, th_, val);
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

template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::abs(const cpptorch::Tensor<T,C> &t)
{
    cpptorch::Tensor<T,C> out(true);
    cpptorch::th::Tensor<T,C>::abs(out, t);
    return out;
}
