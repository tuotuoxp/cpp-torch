#pragma once
#include "../../include/torch/Tensor.h"
#include "TensorPrint.h.inl"
#include "../util.h"


void specifyFully(std::vector<long> &sto_size, int nElements);


template<class TTensor>
cpptorch::Tensor<TTensor>::Tensor(bool auto_create) : th_(nullptr)
{
    if (auto_create)
    {
        create();
    }
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator =(const cpptorch::Tensor<TTensor> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Tensor<TTensor>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Tensor<TTensor>::retain(th_);
        }
    }
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator =(Tensor<TTensor> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Tensor<TTensor>::release(th_);
        th_ = nullptr;
    }
    if (other.th_)
    {
        th_ = other.th_;
        other.th_ = nullptr;
    }
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>::~Tensor()
{
    if (th_)
    {
        cpptorch::th::Tensor<TTensor>::release(th_);
        th_ = nullptr;
    }
}

template<class TTensor>
const std::string cpptorch::Tensor<TTensor>::name() const
{
    if (std::is_same<TTensor, TensorLong>::value)
    {
        return "torch.LongTensor";
    }
    else if (std::is_same<TTensor, TensorFloat>::value)
    {
        return "torch.FloatTensor";
    }
    return "torch.Tensor";
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
void cpptorch::Tensor<TTensor>::create()
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Tensor<TTensor>::create();
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::create(const Storage<typename TTensor::Storage> &storage, long storage_offset,
    const Storage<typename TTensor::SizeStorage> &size)
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Tensor<TTensor>::newWithStorage(storage, storage_offset, size, nullptr);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::create(const Storage<typename TTensor::Storage> &storage, long storage_offset,
    const Storage<typename TTensor::SizeStorage> &size, const Storage<typename TTensor::SizeStorage> &stride)
{
    assert(th_ == nullptr);
    th_ = cpptorch::th::Tensor<TTensor>::newWithStorage(storage, storage_offset, size, stride);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::resize(const Storage<typename TTensor::SizeStorage> &size)
{
    cpptorch::th::Tensor<TTensor>::resize(th_, size, nullptr);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::resize(const Storage<typename TTensor::SizeStorage> &size, const Storage<typename TTensor::SizeStorage> &stride)
{
    cpptorch::th::Tensor<TTensor>::resize(th_, size, stride);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::resizeAs(const Tensor<TTensor> &src)
{
    cpptorch::th::Tensor<TTensor>::resizeAs(th_, src.th_);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::copy(const Tensor<TTensor> &src)
{
    cpptorch::th::Tensor<TTensor>::copy(th_, src.th_);
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
cpptorch::Storage<typename TTensor::Storage> cpptorch::Tensor<TTensor>::storage() const
{
    return cpptorch::Storage<typename TTensor::Storage>(th_ ? cpptorch::th::Tensor<TTensor>::storage(th_) : nullptr);
}

template<class TTensor>
long cpptorch::Tensor<TTensor>::storageOffset() const
{
    return th_ ? cpptorch::th::Tensor<TTensor>::storageOffset(th_) : 0;
}

template<class TTensor>
std::vector<long> cpptorch::Tensor<TTensor>::size() const
{
    std::vector<long> v;
    if (th_)
    {
        typename TTensor::SizeStorage::TH *th = cpptorch::th::Tensor<TTensor>::size(th_);
        long *p = cpptorch::th::Storage<typename TTensor::SizeStorage>::data(th);
        v.assign(p, p + dim());
        cpptorch::th::Storage<typename TTensor::SizeStorage>::release(th);
    }
    return v;
}

template<class TTensor>
long cpptorch::Tensor<TTensor>::size(int dim) const
{
    return cpptorch::th::Tensor<TTensor>::size(th_, dim);
}

template<class TTensor>
std::vector<long> cpptorch::Tensor<TTensor>::stride() const
{
    std::vector<long> v;
    if (th_)
    {
        typename TTensor::SizeStorage::TH *th = cpptorch::th::Tensor<TTensor>::stride(th_);
        long *p = cpptorch::th::Storage<typename TTensor::SizeStorage>::data(th);
        v.assign(p, p + dim());
        cpptorch::th::Storage<typename TTensor::SizeStorage>::release(th);
    }
    return v;
}

template<class TTensor>
int cpptorch::Tensor<TTensor>::dim() const
{
    return cpptorch::th::Tensor<TTensor>::nDimension(th_);
}

template<class TTensor>
typename TTensor::Storage::Base* cpptorch::Tensor<TTensor>::data() const
{
    return cpptorch::th::Tensor<TTensor>::data(th_);
}

template<class TTensor>
cpptorch::Tensor<TTensor>::operator typename TTensor::Storage::Base() const
{
    asserter(nElement() == 1) << "only an 1-D tensor with ONE element can be cast to number";
    return data()[0];
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
bool cpptorch::Tensor<TTensor>::isContiguous() const
{
    return cpptorch::th::Tensor<TTensor>::isContiguous(th_) != 0;
}

template<class TTensor>
int cpptorch::Tensor<TTensor>::nElement() const
{
    return cpptorch::th::Tensor<TTensor>::nElement(th_);
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::narrow(int dimension, long firstIndex, long size) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::narrow(out, th_, dimension, firstIndex, size);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::select(int dimension, long sliceIndex) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::select(out, th_, dimension, sliceIndex);
    return out;
}

template<class TTensor>
template<class TIterator>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::view(const TIterator begin, const TIterator end) const
{
    std::vector<long> sz(begin, end);
    int origNElement = nElement();
    specifyFully(sz, origNElement);

    std::vector<long> ss = size();
    assert(isContiguous() && "expecting a contiguous tensor");
    cpptorch::Tensor<TTensor> view;
    view.create(storage(), storageOffset(), sz);
    assert(view.nElement() == origNElement && "Wrong size for view. ");
    return view;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::expand(const std::vector<long> &sz) const
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
    cpptorch::Tensor<TTensor> output;
    output.create(storage(), storageOffset(), tensor_size, tensor_stride);
    return output;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::t() const
{
    cpptorch::Tensor<TTensor> output(true);
    cpptorch::th::Tensor<TTensor>::transpose(output, th_, 0, 1);
    return output;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::operator [] (const std::initializer_list<long> &inputs) const
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

    cpptorch::Tensor<TTensor> output;
    if (tsize.size() == 0)
    {
        tsize.push_back(1);
        tstride.push_back(1);
    }
    output.create(storage(), index, cpptorch::Storage<StorageLong>(tsize), cpptorch::Storage<StorageLong>(tstride));
    return output;
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
void cpptorch::Tensor<TTensor>::fill(typename TTensor::Storage::Base val)
{
    cpptorch::th::Tensor<TTensor>::fill(th_, val);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::abs()
{
    cpptorch::th::Tensor<TTensor>::abs(th_, th_);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::addmv(typename TTensor::Storage::Base beta, const Tensor<TTensor> &t,
    typename TTensor::Storage::Base alpha, const Tensor<TTensor> &matrix, const Tensor<TTensor> &vector)
{
    cpptorch::th::Tensor<TTensor>::addmv(th_, beta, t, alpha, matrix, vector);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::addmm(typename TTensor::Storage::Base beta, const Tensor<TTensor> &t,
    typename TTensor::Storage::Base alpha, const Tensor<TTensor> &matrix1, const Tensor<TTensor> &matrix2)
{
    cpptorch::th::Tensor<TTensor>::addmm(th_, beta, t, alpha, matrix1, matrix2);
}

template<class TTensor>
void cpptorch::Tensor<TTensor>::addr(typename TTensor::Storage::Base beta, const Tensor<TTensor> &t,
    typename TTensor::Storage::Base alpha, const Tensor<TTensor> &vector1, const Tensor<TTensor> &vector2)
{
    cpptorch::th::Tensor<TTensor>::addr(th_, beta, t, alpha, vector1, vector2);
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator += (typename TTensor::Storage::Base val)
{
    cpptorch::th::Tensor<TTensor>::add(th_, th_, val);
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator += (const cpptorch::Tensor<TTensor> &other)
{
    cpptorch::th::Tensor<TTensor>::cadd(th_, th_, 1, other);
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator -= (const cpptorch::Tensor<TTensor> &other)
{
    cpptorch::th::Tensor<TTensor>::cadd(th_, th_, -1, other);
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator *= (typename TTensor::Storage::Base val)
{
    cpptorch::th::Tensor<TTensor>::mul(th_, th_, val);
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator *= (const cpptorch::Tensor<TTensor> &other)
{
    cpptorch::th::Tensor<TTensor>::cmul(th_, th_, other);
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator ^= (typename TTensor::Storage::Base val)
{
    cpptorch::th::Tensor<TTensor>::pow(th_, th_, val);
    return *this;
}

template<class TTensor>
cpptorch::Tensor<TTensor>& cpptorch::Tensor<TTensor>::operator ^= (const Tensor<TTensor> &other)
{
    cpptorch::th::Tensor<TTensor>::cpow(th_, th_, other);
    return *this;
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
typename TTensor::Storage::Base cpptorch::Tensor<TTensor>::minall() const
{
    return th_ ? cpptorch::th::Tensor<TTensor>::minall(th_) : 0;
}

template<class TTensor>
typename TTensor::Storage::Base cpptorch::Tensor<TTensor>::maxall() const
{
    return th_ ? cpptorch::th::Tensor<TTensor>::maxall(th_) : 0;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::max(int dimension) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::max(out.th_, th_, dimension);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::sum(int dimension) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::sum(out.th_, th_, dimension);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::operator +(typename TTensor::Storage::Base val) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::add(out.th_, th_, val);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::operator +(const Tensor<TTensor> &other) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::cadd(out, th_, 1, other);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::operator -(typename TTensor::Storage::Base val) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::add(out, th_, -val);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::operator -(const Tensor<TTensor> &other) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::cadd(out, th_, (typename TTensor::Storage::Base)-1.0, other);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::operator /(const Tensor<TTensor> &other) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::cdiv(out, th_, other);
    return out;
}

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::Tensor<TTensor>::operator ^(typename TTensor::Storage::Base val) const
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::pow(out, th_, val);
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

template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::abs(const cpptorch::Tensor<TTensor> &t)
{
    cpptorch::Tensor<TTensor> out(true);
    cpptorch::th::Tensor<TTensor>::abs(out, t);
    return out;
}
