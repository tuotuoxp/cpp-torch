#pragma once
#include "Tensor.h"
#include "TensorPrint.h"
#include "../th_wrapper.h"
#include "../util.h"
#include "../extractor.h"


void specifyFully(std::vector<long> &sto_size, int nElements);


template<class TTensor>
nn::Tensor<TTensor>::Tensor(bool auto_create) : th_(nullptr)
{
    if (auto_create)
    {
        create();
    }
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator =(const nn::Tensor<TTensor> &other)
{
    if (this != &other) {
        if (th_)
        {
            THWrapper::Tensor<TTensor>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            THWrapper::Tensor<TTensor>::retain(th_);
        }
    }
    return *this;
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator =(Tensor<TTensor> &&other)
{
    assert(this != &other);
    if (th_)
    {
        THWrapper::Tensor<TTensor>::release(th_);
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
nn::Tensor<TTensor>::~Tensor()
{
    if (th_)
    {
        THWrapper::Tensor<TTensor>::release(th_);
        th_ = nullptr;
    }
}

template<class TTensor>
const std::string nn::Tensor<TTensor>::name() const
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
void nn::Tensor<TTensor>::create()
{
    assert(th_ == nullptr);
    th_ = THWrapper::Tensor<TTensor>::create();
}

template<class TTensor>
void nn::Tensor<TTensor>::create(const Storage<typename TTensor::Storage> &storage, long storage_offset,
    const Storage<typename TTensor::SizeStorage> &size)
{
    assert(th_ == nullptr);
    th_ = THWrapper::Tensor<TTensor>::newWithStorage(storage, storage_offset, size, nullptr);
}

template<class TTensor>
void nn::Tensor<TTensor>::create(const Storage<typename TTensor::Storage> &storage, long storage_offset,
    const Storage<typename TTensor::SizeStorage> &size, const Storage<typename TTensor::SizeStorage> &stride)
{
    assert(th_ == nullptr);
    th_ = THWrapper::Tensor<TTensor>::newWithStorage(storage, storage_offset, size, stride);
}

template<class TTensor>
void nn::Tensor<TTensor>::resize(const Storage<typename TTensor::SizeStorage> &size)
{
    THWrapper::Tensor<TTensor>::resize(th_, size, nullptr);
}

template<class TTensor>
void nn::Tensor<TTensor>::resize(const Storage<typename TTensor::SizeStorage> &size, const Storage<typename TTensor::SizeStorage> &stride)
{
    THWrapper::Tensor<TTensor>::resize(th_, size, stride);
}

template<class TTensor>
void nn::Tensor<TTensor>::resizeAs(const Tensor<TTensor> &src)
{
    THWrapper::Tensor<TTensor>::resizeAs(th_, src.th_);
}

template<class TTensor>
void nn::Tensor<TTensor>::copy(const Tensor<TTensor> &src)
{
    THWrapper::Tensor<TTensor>::copy(th_, src.th_);
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
nn::Storage<typename TTensor::Storage> nn::Tensor<TTensor>::storage() const
{
    return nn::Storage<typename TTensor::Storage>(th_ ? THWrapper::Tensor<TTensor>::storage(th_) : nullptr);
}

template<class TTensor>
long nn::Tensor<TTensor>::storageOffset() const
{
    return th_ ? THWrapper::Tensor<TTensor>::storageOffset(th_) : 0;
}

template<class TTensor>
std::vector<long> nn::Tensor<TTensor>::size() const
{
    std::vector<long> v;
    if (th_)
    {
        typename TTensor::SizeStorage::THStorage *th = THWrapper::Tensor<TTensor>::size(th_);
        long *p = THWrapper::Storage<typename TTensor::SizeStorage>::data(th);
        v.assign(p, p + dim());
        THWrapper::Storage<typename TTensor::SizeStorage>::release(th);
    }
    return v;
}

template<class TTensor>
long nn::Tensor<TTensor>::size(int dim) const
{
    return THWrapper::Tensor<TTensor>::size(th_, dim);
}

template<class TTensor>
std::vector<long> nn::Tensor<TTensor>::stride() const
{
    std::vector<long> v;
    if (th_)
    {
        typename TTensor::SizeStorage::THStorage *th = THWrapper::Tensor<TTensor>::stride(th_);
        long *p = THWrapper::Storage<typename TTensor::SizeStorage>::data(th);
        v.assign(p, p + dim());
        THWrapper::Storage<typename TTensor::SizeStorage>::release(th);
    }
    return v;
}

template<class TTensor>
int nn::Tensor<TTensor>::dim() const
{
    return THWrapper::Tensor<TTensor>::nDimension(th_);
}

template<class TTensor>
typename TTensor::Storage::StorageBase* nn::Tensor<TTensor>::data() const
{
    return THWrapper::Tensor<TTensor>::data(th_);
}

template<class TTensor>
nn::Tensor<TTensor>::operator typename TTensor::Storage::StorageBase() const
{
    asserter(nElement() == 1) << "only an 1-D tensor with ONE element can be cast to number";
    return data()[0];
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
bool nn::Tensor<TTensor>::isContiguous() const
{
    return THWrapper::Tensor<TTensor>::isContiguous(th_) != 0;
}

template<class TTensor>
int nn::Tensor<TTensor>::nElement() const
{
    return THWrapper::Tensor<TTensor>::nElement(th_);
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::narrow(int dimension, long firstIndex, long size) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::narrow(out, th_, dimension, firstIndex, size);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::select(int dimension, long sliceIndex) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::select(out, th_, dimension, sliceIndex);
    return out;
}

template<class TTensor>
template<class TIterator>
nn::Tensor<TTensor> nn::Tensor<TTensor>::view(const TIterator begin, const TIterator end) const
{
    std::vector<long> sz(begin, end);
    int origNElement = nElement();
    specifyFully(sz, origNElement);

    std::vector<long> ss = size();
    assert(isContiguous() && "expecting a contiguous tensor");
    nn::Tensor<TTensor> view;
    view.create(storage(), storageOffset(), sz);
    assert(view.nElement() == origNElement && "Wrong size for view. ");
    return view;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::expand(const std::vector<long> &sz) const
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
    nn::Tensor<TTensor> output;
    output.create(storage(), storageOffset(), tensor_size, tensor_stride);
    return output;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::t() const
{
    nn::Tensor<TTensor> output(true);
    THWrapper::Tensor<TTensor>::transpose(output, th_, 0, 1);
    return output;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::operator [] (const std::initializer_list<long> &inputs) const
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

    nn::Tensor<TTensor> output;
    if (tsize.size() == 0)
    {
        tsize.push_back(1);
        tstride.push_back(1);
    }
    output.create(storage(), index, nn::Storage<StorageLong>(tsize), nn::Storage<StorageLong>(tstride));
    return output;
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
void nn::Tensor<TTensor>::fill(typename TTensor::Storage::StorageBase val)
{
    THWrapper::Tensor<TTensor>::fill(th_, val);
}

template<class TTensor>
void nn::Tensor<TTensor>::abs()
{
    THWrapper::Tensor<TTensor>::abs(th_, th_);
}

template<class TTensor>
void nn::Tensor<TTensor>::addmv(typename TTensor::Storage::StorageBase beta, const Tensor<TTensor> &t,
    typename TTensor::Storage::StorageBase alpha, const Tensor<TTensor> &matrix, const Tensor<TTensor> &vector)
{
    THWrapper::Tensor<TTensor>::addmv(th_, beta, t, alpha, matrix, vector);
}

template<class TTensor>
void nn::Tensor<TTensor>::addmm(typename TTensor::Storage::StorageBase beta, const Tensor<TTensor> &t,
    typename TTensor::Storage::StorageBase alpha, const Tensor<TTensor> &matrix1, const Tensor<TTensor> &matrix2)
{
    THWrapper::Tensor<TTensor>::addmm(th_, beta, t, alpha, matrix1, matrix2);
}

template<class TTensor>
void nn::Tensor<TTensor>::addr(typename TTensor::Storage::StorageBase beta, const Tensor<TTensor> &t,
    typename TTensor::Storage::StorageBase alpha, const Tensor<TTensor> &vector1, const Tensor<TTensor> &vector2)
{
    THWrapper::Tensor<TTensor>::addr(th_, beta, t, alpha, vector1, vector2);
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator += (typename TTensor::Storage::StorageBase val)
{
    THWrapper::Tensor<TTensor>::add(th_, th_, val);
    return *this;
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator += (const nn::Tensor<TTensor> &other)
{
    THWrapper::Tensor<TTensor>::cadd(th_, th_, 1, other);
    return *this;
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator -= (const nn::Tensor<TTensor> &other)
{
    THWrapper::Tensor<TTensor>::cadd(th_, th_, -1, other);
    return *this;
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator *= (typename TTensor::Storage::StorageBase val)
{
    THWrapper::Tensor<TTensor>::mul(th_, th_, val);
    return *this;
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator *= (const nn::Tensor<TTensor> &other)
{
    THWrapper::Tensor<TTensor>::cmul(th_, th_, other);
    return *this;
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator ^= (typename TTensor::Storage::StorageBase val)
{
    THWrapper::Tensor<TTensor>::pow(th_, th_, val);
    return *this;
}

template<class TTensor>
nn::Tensor<TTensor>& nn::Tensor<TTensor>::operator ^= (const Tensor<TTensor> &other)
{
    THWrapper::Tensor<TTensor>::cpow(th_, th_, other);
    return *this;
}

//////////////////////////////////////////////////////////////////////////

template<class TTensor>
typename TTensor::Storage::StorageBase nn::Tensor<TTensor>::minall() const
{
    return th_ ? THWrapper::Tensor<TTensor>::minall(th_) : 0;
}

template<class TTensor>
typename TTensor::Storage::StorageBase nn::Tensor<TTensor>::maxall() const
{
    return th_ ? THWrapper::Tensor<TTensor>::maxall(th_) : 0;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::max(int dimension) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::max(out.th_, th_, dimension);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::sum(int dimension) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::sum(out.th_, th_, dimension);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::operator +(typename TTensor::Storage::StorageBase val) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::add(out.th_, th_, val);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::operator +(const Tensor<TTensor> &other) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::cadd(out, th_, 1, other);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::operator -(typename TTensor::Storage::StorageBase val) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::add(out, th_, -val);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::operator -(const Tensor<TTensor> &other) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::cadd(out, th_, -1.0, other);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::operator /(const Tensor<TTensor> &other) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::cdiv(out, th_, other);
    return out;
}

template<class TTensor>
nn::Tensor<TTensor> nn::Tensor<TTensor>::operator ^(typename TTensor::Storage::StorageBase val) const
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::pow(out, th_, val);
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

template<class TTensor>
std::ostream& operator << (std::ostream &o, const nn::Tensor<TTensor> &m)
{
    TensorPrint<TensorFloat>(o, m).printTensor();
    o << std::endl;
    return o;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<class TTensor>
nn::Tensor<TTensor> nn::abs(const nn::Tensor<TTensor> &t)
{
    nn::Tensor<TTensor> out(true);
    THWrapper::Tensor<TTensor>::abs(out, t);
    return out;
}
