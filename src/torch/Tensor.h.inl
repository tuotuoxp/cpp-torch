#pragma once
#include "../../include/torch/Tensor.h"
#include "TensorPrint.h.inl"
#include "../util.h"


void specifyFully(std::vector<long> &sto_size, int nElements);


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>::Tensor(bool auto_create) : th_(nullptr)
{
    if (auto_create)
    {
        create();
    }
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator =(const cpptorch::Tensor<T, F> &other)
{
    if (this != &other) {
        if (th_)
        {
            cpptorch::th::Tensor<T, F>::release(th_);
            th_ = nullptr;
        }
        if (other.th_)
        {
            th_ = other.th_;
            cpptorch::th::Tensor<T, F>::retain(th_);
        }
    }
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator =(Tensor<T, F> &&other)
{
    assert(this != &other);
    if (th_)
    {
        cpptorch::th::Tensor<T, F>::release(th_);
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
cpptorch::Tensor<T, F>::~Tensor()
{
    if (th_)
    {
        cpptorch::th::Tensor<T, F>::release(th_);
        th_ = nullptr;
    }
}

template<typename T, GPUFlag F>
const std::string cpptorch::Tensor<T, F>::name() const
{
    if (F == GPU_Cuda)
    {
        return "torch.CudaTensor";
    }
    else if (std::is_same<T, long>::value)
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

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::create()
{
    assert(th_ == nullptr);
    Storage<T, F> s;
    s.create();
    th_ = cpptorch::th::Tensor<T, F>::newWithStorage(s, 0, 0, nullptr, nullptr);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::create(const Storage<T, F> &storage, long storage_offset, int dim,
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
    th_ = cpptorch::th::Tensor<T, F>::newWithStorage(storage, storage_offset, dim, size, stride);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::resize(const std::vector<long> &size)
{
    cpptorch::Storage<long, GPU_None> sz(size);
    cpptorch::th::Tensor<T, F>::resize(th_, sz, nullptr);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::resize(const std::vector<long> &size, const std::vector<long> &stride)
{
    cpptorch::Storage<long, GPU_None> sz(size);
    cpptorch::Storage<long, GPU_None> st(stride);
    cpptorch::th::Tensor<T, F>::resize(th_, sz, st);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::resizeAs(const Tensor<T, F> &src)
{
    cpptorch::th::Tensor<T, F>::resizeAs(th_, src.th_);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::copy(const Tensor<T, F> &src)
{
    cpptorch::th::Tensor<T, F>::copy(th_, src.th_);
}

//////////////////////////////////////////////////////////////////////////

template<typename T, GPUFlag F>
cpptorch::Storage<T, F> cpptorch::Tensor<T, F>::storage() const
{
    return cpptorch::Storage<T, F>(th_ ? cpptorch::th::Tensor<T, F>::storage(th_) : nullptr);
}

template<typename T, GPUFlag F>
long cpptorch::Tensor<T, F>::storageOffset() const
{
    return th_ ? cpptorch::th::Tensor<T, F>::storageOffset(th_) : 0;
}

template<typename T, GPUFlag F>
std::vector<long> cpptorch::Tensor<T, F>::size() const
{
    std::vector<long> v;
    if (th_)
    {
        Storage<long, GPU_None> sz(cpptorch::th::Tensor<T, F>::size(th_));
        long *p = sz.data();
        v.assign(p, p + dim());
    }
    return v;
}

template<typename T, GPUFlag F>
long cpptorch::Tensor<T, F>::size(int dim) const
{
    return cpptorch::th::Tensor<T, F>::size(th_, dim);
}

template<typename T, GPUFlag F>
std::vector<long> cpptorch::Tensor<T, F>::stride() const
{
    std::vector<long> v;
    if (th_)
    {
        THTrait<long, GPU_None>::Storage *th = cpptorch::th::Tensor<T, F>::stride(th_);
        long *p = cpptorch::th::Storage<long, GPU_None>::data(th);
        v.assign(p, p + dim());
        cpptorch::th::Storage<long, GPU_None>::release(th);
    }
    return v;
}

template<typename T, GPUFlag F>
int cpptorch::Tensor<T, F>::dim() const
{
    return cpptorch::th::Tensor<T, F>::nDimension(th_);
}

template<typename T, GPUFlag F>
T* cpptorch::Tensor<T, F>::data() const
{
    return cpptorch::th::Tensor<T, F>::data(th_);
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>::operator T() const
{
    asserter(nElement() == 1) << "only an 1-D tensor with ONE element can be cast to number";
    return cpptorch::th::Storage<T, F>::data_by_index(cpptorch::th::Tensor<T, F>::storage(th_), storageOffset());
}

//////////////////////////////////////////////////////////////////////////

template<typename T, GPUFlag F>
bool cpptorch::Tensor<T, F>::isContiguous() const
{
    return cpptorch::th::Tensor<T, F>::isContiguous(th_) != 0;
}

template<typename T, GPUFlag F>
int cpptorch::Tensor<T, F>::nElement() const
{
    return cpptorch::th::Tensor<T, F>::nElement(th_);
}

//////////////////////////////////////////////////////////////////////////

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::narrow(int dimension, long firstIndex, long size) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::narrow(out, th_, dimension, firstIndex, size);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::select(int dimension, long sliceIndex) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::select(out, th_, dimension, sliceIndex);
    return out;
}

template<typename T, GPUFlag F>
template<class TIterator>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::view(const TIterator begin, const TIterator end) const
{
    std::vector<long> sz(begin, end);
    int origNElement = nElement();
    specifyFully(sz, origNElement);

    std::vector<long> ss = size();
    assert(isContiguous() && "expecting a contiguous tensor");
    cpptorch::Tensor<T, F> view;
    view.create(storage(), storageOffset(), (int)sz.size(), &sz[0]);
    assert(view.nElement() == origNElement && "Wrong size for view. ");
    return view;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::expand(const std::vector<long> &sz) const
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
    cpptorch::Tensor<T, F> output;
    output.create(storage(), storageOffset(), tensor_dim, &tensor_size[0], &tensor_stride[0]);
    return output;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::t() const
{
    cpptorch::Tensor<T, F> output(true);
    cpptorch::th::Tensor<T, F>::transpose(output, th_, 0, 1);
    return output;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::operator [] (const std::initializer_list<long> &inputs) const
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

    cpptorch::Tensor<T, F> output;
    if (tsize.size() == 0)
    {
        tsize.push_back(1);
        tstride.push_back(1);
    }
    output.create(storage(), index, (int)tsize.size(), &tsize[0], &tstride[0]);
    return output;
}

//////////////////////////////////////////////////////////////////////////

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::fill(T val)
{
    cpptorch::th::Tensor<T, F>::fill(th_, val);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::abs()
{
    cpptorch::th::Tensor<T, F>::abs(th_, th_);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::addmv(T beta, const Tensor<T, F> &t,
    T alpha, const Tensor<T, F> &matrix, const Tensor<T, F> &vector)
{
    cpptorch::th::Tensor<T, F>::addmv(th_, beta, t, alpha, matrix, vector);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::addmm(T beta, const Tensor<T, F> &t,
    T alpha, const Tensor<T, F> &matrix1, const Tensor<T, F> &matrix2)
{
    cpptorch::th::Tensor<T, F>::addmm(th_, beta, t, alpha, matrix1, matrix2);
}

template<typename T, GPUFlag F>
void cpptorch::Tensor<T, F>::addr(T beta, const Tensor<T, F> &t,
    T alpha, const Tensor<T, F> &vector1, const Tensor<T, F> &vector2)
{
    cpptorch::th::Tensor<T, F>::addr(th_, beta, t, alpha, vector1, vector2);
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator += (T val)
{
    cpptorch::th::Tensor<T, F>::add(th_, th_, val);
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator += (const cpptorch::Tensor<T, F> &other)
{
    cpptorch::th::Tensor<T, F>::cadd(th_, th_, 1, other);
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator -= (const cpptorch::Tensor<T, F> &other)
{
    cpptorch::th::Tensor<T, F>::cadd(th_, th_, -1, other);
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator *= (T val)
{
    cpptorch::th::Tensor<T, F>::mul(th_, th_, val);
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator *= (const cpptorch::Tensor<T, F> &other)
{
    cpptorch::th::Tensor<T, F>::cmul(th_, th_, other);
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator ^= (T val)
{
    cpptorch::th::Tensor<T, F>::pow(th_, th_, val);
    return *this;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F>& cpptorch::Tensor<T, F>::operator ^= (const Tensor<T, F> &other)
{
    cpptorch::th::Tensor<T, F>::cpow(th_, th_, other);
    return *this;
}

//////////////////////////////////////////////////////////////////////////

template<typename T, GPUFlag F>
T cpptorch::Tensor<T, F>::minall() const
{
    return th_ ? cpptorch::th::Tensor<T, F>::minall(th_) : 0;
}

template<typename T, GPUFlag F>
T cpptorch::Tensor<T, F>::maxall() const
{
    return th_ ? cpptorch::th::Tensor<T, F>::maxall(th_) : 0;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::max(int dimension) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::max(out.th_, th_, dimension);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::sum(int dimension) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::sum(out.th_, th_, dimension);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::operator +(T val) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::add(out.th_, th_, val);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::operator +(const Tensor<T, F> &other) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::cadd(out, th_, 1, other);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::operator -(T val) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::add(out, th_, -val);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::operator -(const Tensor<T, F> &other) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::cadd(out, th_, (T)-1.0, other);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::operator /(const Tensor<T, F> &other) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::cdiv(out, th_, other);
    return out;
}

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::Tensor<T, F>::operator ^(T val) const
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::pow(out, th_, val);
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

template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::abs(const cpptorch::Tensor<T, F> &t)
{
    cpptorch::Tensor<T, F> out(true);
    cpptorch::th::Tensor<T, F>::abs(out, t);
    return out;
}
