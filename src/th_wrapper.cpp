#include "th_wrapper.h"
#include "../include/torch/Storage.h"
#include "../include/torch/Tensor.h"
#include "allocator.h"

#include <TH/TH.h>
#include <THNN/THNN.h>
#include <assert.h>


template<typename T>
static T* bypass(const T *ptr_src, long count, bool take_ownership_of_data)
{
    if (!take_ownership_of_data)
    {
        long sz = count * sizeof(T);
        T *ptr = (T*)malloc(sz);
        memcpy(ptr, ptr_src, sz);
        return const_cast<T*>(ptr);
    }
    else
    {
        return const_cast<T*>(ptr_src);
    }
}


namespace cpptorch { namespace th {

template<>
THLongStorage* Storage<long, GPU_None>::newWithData(const long *ptr_src, long count, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        return THLongStorage_newWithDataAndAllocator(bypass(ptr_src, count, take_ownership_of_data), count,
            cpptorch::allocator::get(), cpptorch::allocator::requestIndex(count * sizeof(long)));
    }
    else
    {
        return THLongStorage_newWithAllocator(0, cpptorch::allocator::get(), cpptorch::allocator::requestIndex(0));
    }
}
template<>
THFloatStorage* Storage<float, GPU_None>::newWithData(const float *ptr_src, long count, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        return THFloatStorage_newWithDataAndAllocator(bypass(ptr_src, count, take_ownership_of_data), count,
            cpptorch::allocator::get(), cpptorch::allocator::requestIndex(count * sizeof(float)));
    }
    else
    {
        return THFloatStorage_newWithAllocator(0, cpptorch::allocator::get(), cpptorch::allocator::requestIndex(0));
    }
}
template<>
THDoubleStorage* Storage<double, GPU_None>::newWithData(const double *ptr_src, long count, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        return THDoubleStorage_newWithDataAndAllocator(bypass(ptr_src, count, take_ownership_of_data), count,
            cpptorch::allocator::get(), cpptorch::allocator::requestIndex(count * sizeof(double)));
    }
    else
    {
        return THDoubleStorage_newWithAllocator(0, cpptorch::allocator::get(), cpptorch::allocator::requestIndex(0));
    }
}

template<>
void Storage<long, GPU_None>::retain(THLongStorage *storage)
{
    THLongStorage_retain(storage);
}
template<>
void Storage<float, GPU_None>::retain(THFloatStorage *storage)
{
    THFloatStorage_retain(storage);
}
template<>
void Storage<double, GPU_None>::retain(THDoubleStorage *storage)
{
    THDoubleStorage_retain(storage);
}

template<>
void Storage<long, GPU_None>::release(THLongStorage *storage)
{
    THLongStorage_free(storage);
}
template<>
void Storage<float, GPU_None>::release(THFloatStorage *storage)
{
    THFloatStorage_free(storage);
}
template<>
void Storage<double, GPU_None>::release(THDoubleStorage *storage)
{
    THDoubleStorage_free(storage);
}

//////////////////////////////////////////////////////////////////////////

template<>
long* Storage<long, GPU_None>::data(THLongStorage *storage)
{
    return THLongStorage_data(storage);
}
template<>
float* Storage<float, GPU_None>::data(THFloatStorage *storage)
{
    return THFloatStorage_data(storage);
}
template<>
double* Storage<double, GPU_None>::data(THDoubleStorage *storage)
{
    return THDoubleStorage_data(storage);
}

template<>
long Storage<long, GPU_None>::data_by_index(const THLongStorage *storage, long index)
{
    return THLongStorage_get(storage, index);
}
template<>
float Storage<float, GPU_None>::data_by_index(const THFloatStorage *storage, long index)
{
    return THFloatStorage_get(storage, index);
}
template<>
double Storage<double, GPU_None>::data_by_index(const THDoubleStorage *storage, long index)
{
    return THDoubleStorage_get(storage, index);
}

template<>
long Storage<long, GPU_None>::size(const THLongStorage *storage)
{
    return (long)THLongStorage_size(storage);
}
template<>
long Storage<float, GPU_None>::size(const THFloatStorage *storage)
{
    return (long)THFloatStorage_size(storage);
}
template<>
long Storage<double, GPU_None>::size(const THDoubleStorage *storage)
{
    return (long)THDoubleStorage_size(storage);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
THLongTensor* Tensor<long, GPU_None>::newWithStorage(THLongStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THLongTensor_newWithStorage(storage, offset, nullptr, nullptr);
    case 1: return THLongTensor_newWithStorage1d(storage, offset, size[0], stride[0]);
    case 2: return THLongTensor_newWithStorage2d(storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THLongTensor_newWithStorage3d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THLongTensor_newWithStorage4d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THLongTensor_newWithStorage(storage, offset, cpptorch::Storage<long, GPU_None>(size, dim, false),
        cpptorch::Storage<long, GPU_None>(stride, dim, false));
}
template<>
THFloatTensor* Tensor<float, GPU_None>::newWithStorage(THFloatStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THFloatTensor_newWithStorage(storage, offset, nullptr, nullptr);
    case 1: return THFloatTensor_newWithStorage1d(storage, offset, size[0], stride[0]);
    case 2: return THFloatTensor_newWithStorage2d(storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THFloatTensor_newWithStorage3d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THFloatTensor_newWithStorage4d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THFloatTensor_newWithStorage(storage, offset, cpptorch::Storage<long, GPU_None>(size, dim, false),
        cpptorch::Storage<long, GPU_None>(stride, dim, false));
}
template<>
THDoubleTensor* Tensor<double, GPU_None>::newWithStorage(THDoubleStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THDoubleTensor_newWithStorage(storage, offset, nullptr, nullptr);
    case 1: return THDoubleTensor_newWithStorage1d(storage, offset, size[0], stride[0]);
    case 2: return THDoubleTensor_newWithStorage2d(storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THDoubleTensor_newWithStorage3d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THDoubleTensor_newWithStorage4d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THDoubleTensor_newWithStorage(storage, offset, cpptorch::Storage<long, GPU_None>(size, dim, false),
        cpptorch::Storage<long, GPU_None>(stride, dim, false));
}

template<>
void Tensor<long, GPU_None>::resize(THLongTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THLongTensor_resize(self, size, stride);
}
template<>
void Tensor<float, GPU_None>::resize(THFloatTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THFloatTensor_resize(self, size, stride);
}
template<>
void Tensor<double, GPU_None>::resize(THDoubleTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THDoubleTensor_resize(self, size, stride);
}

template<>
void Tensor<long, GPU_None>::resizeAs(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_resizeAs(self, src);
}
template<>
void Tensor<float, GPU_None>::resizeAs(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_resizeAs(self, src);
}
template<>
void Tensor<double, GPU_None>::resizeAs(THDoubleTensor *self, THDoubleTensor *src)
{
    THDoubleTensor_resizeAs(self, src);
}

template<>
void Tensor<long, GPU_None>::copy(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_copy(self, src);
}
template<>
void Tensor<float, GPU_None>::copy(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_copy(self, src);
}
template<>
void Tensor<double, GPU_None>::copy(THDoubleTensor *self, THDoubleTensor *src)
{
    THDoubleTensor_copy(self, src);
}

template<>
void Tensor<long, GPU_None>::retain(THLongTensor *tensor)
{
    THLongTensor_retain(tensor);
}
template<>
void Tensor<float, GPU_None>::retain(THFloatTensor *tensor)
{
    THFloatTensor_retain(tensor);
}
template<>
void Tensor<double, GPU_None>::retain(THDoubleTensor *tensor)
{
    THDoubleTensor_retain(tensor);
}

template<>
void Tensor<long, GPU_None>::release(THLongTensor *tensor)
{
    THLongTensor_free(tensor);
}
template<>
void Tensor<float, GPU_None>::release(THFloatTensor *tensor)
{
    THFloatTensor_free(tensor);
}
template<>
void Tensor<double, GPU_None>::release(THDoubleTensor *tensor)
{
    THDoubleTensor_free(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
THLongStorage* Tensor<long, GPU_None>::storage(const THLongTensor *tensor)
{
    return THLongTensor_storage(tensor);
}
template<>
THFloatStorage* Tensor<float, GPU_None>::storage(const THFloatTensor *tensor)
{
    return THFloatTensor_storage(tensor);
}
template<>
THDoubleStorage* Tensor<double, GPU_None>::storage(const THDoubleTensor *tensor)
{
    return THDoubleTensor_storage(tensor);
}

template<>
long Tensor<long, GPU_None>::storageOffset(const THLongTensor *tensor)
{
    return (long)THLongTensor_storageOffset(tensor);
}
template<>
long Tensor<float, GPU_None>::storageOffset(const THFloatTensor *tensor)
{
    return (long)THFloatTensor_storageOffset(tensor);
}
template<>
long Tensor<double, GPU_None>::storageOffset(const THDoubleTensor *tensor)
{
    return (long)THDoubleTensor_storageOffset(tensor);
}

template<>
int Tensor<long, GPU_None>::nDimension(const THLongTensor *tensor)
{
    return THLongTensor_nDimension(tensor);
}
template<>
int Tensor<float, GPU_None>::nDimension(const THFloatTensor *tensor)
{
    return THFloatTensor_nDimension(tensor);
}
template<>
int Tensor<double, GPU_None>::nDimension(const THDoubleTensor *tensor)
{
    return THDoubleTensor_nDimension(tensor);
}

template<>
THLongStorage* Tensor<long, GPU_None>::size(const THLongTensor *tensor)
{
    return THLongTensor_newSizeOf((THLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float, GPU_None>::size(const THFloatTensor *tensor)
{
    return THFloatTensor_newSizeOf((THFloatTensor*)tensor);
}
template<>
THLongStorage* Tensor<double, GPU_None>::size(const THDoubleTensor *tensor)
{
    return THDoubleTensor_newSizeOf((THDoubleTensor*)tensor);
}

template<>
long Tensor<long, GPU_None>::size(const THLongTensor *tensor, int dim)
{
    return THLongTensor_size(tensor, dim);
}
template<>
long Tensor<float, GPU_None>::size(const THFloatTensor *tensor, int dim)
{
    return THFloatTensor_size(tensor, dim);
}
template<>
long Tensor<double, GPU_None>::size(const THDoubleTensor *tensor, int dim)
{
    return THDoubleTensor_size(tensor, dim);
}

template<>
THLongStorage* Tensor<long, GPU_None>::stride(const THLongTensor *tensor)
{
    return THLongTensor_newStrideOf((THLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float, GPU_None>::stride(const THFloatTensor *tensor)
{
    return THFloatTensor_newStrideOf((THFloatTensor*)tensor);
}
template<>
THLongStorage* Tensor<double, GPU_None>::stride(const THDoubleTensor *tensor)
{
    return THDoubleTensor_newStrideOf((THDoubleTensor*)tensor);
}

template<>
long *Tensor<long, GPU_None>::data(const THLongTensor *tensor)
{
    return THLongTensor_data((THLongTensor*)tensor);
}
template<>
float *Tensor<float, GPU_None>::data(const THFloatTensor *tensor)
{
    return THFloatTensor_data((THFloatTensor*)tensor);
}
template<>
double *Tensor<double, GPU_None>::data(const THDoubleTensor *tensor)
{
    return THDoubleTensor_data((THDoubleTensor*)tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
int Tensor<long, GPU_None>::isContiguous(const THLongTensor *tensor)
{
    return THLongTensor_isContiguous(tensor);
}
template<>
int Tensor<float, GPU_None>::isContiguous(const THFloatTensor *tensor)
{
    return THFloatTensor_isContiguous(tensor);
}
template<>
int Tensor<double, GPU_None>::isContiguous(const THDoubleTensor *tensor)
{
    return THDoubleTensor_isContiguous(tensor);
}

template<>
int Tensor<long, GPU_None>::isSameSizeAs(const THLongTensor *self, const THLongTensor *src)
{
    return THLongTensor_isSameSizeAs(self, src);
}
template<>
int Tensor<float, GPU_None>::isSameSizeAs(const THFloatTensor *self, const THFloatTensor *src)
{
    return THFloatTensor_isSameSizeAs(self, src);
}
template<>
int Tensor<double, GPU_None>::isSameSizeAs(const THDoubleTensor *self, const THDoubleTensor *src)
{
    return THDoubleTensor_isSameSizeAs(self, src);
}

template<>
long Tensor<long, GPU_None>::nElement(const THLongTensor *tensor)
{
    return (long)THLongTensor_nElement(tensor);
}
template<>
long Tensor<float, GPU_None>::nElement(const THFloatTensor *tensor)
{
    return (long)THFloatTensor_nElement(tensor);
}
template<>
long Tensor<double, GPU_None>::nElement(const THDoubleTensor *tensor)
{
    return (long)THDoubleTensor_nElement(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long, GPU_None>::narrow(THLongTensor *self, THLongTensor *src, int dimension, long firstIndex, long size)
{
    THLongTensor_narrow(self, src, dimension, firstIndex, size);
}
template<>
void Tensor<float, GPU_None>::narrow(THFloatTensor *self, THFloatTensor *src, int dimension, long firstIndex, long size)
{
    THFloatTensor_narrow(self, src, dimension, firstIndex, size);
}
template<>
void Tensor<double, GPU_None>::narrow(THDoubleTensor *self, THDoubleTensor *src, int dimension, long firstIndex, long size)
{
    THDoubleTensor_narrow(self, src, dimension, firstIndex, size);
}

template<>
void Tensor<long, GPU_None>::select(THLongTensor *self, THLongTensor *src, int dimension, long sliceIndex)
{
    THLongTensor_select(self, src, dimension, sliceIndex);
}
template<>
void Tensor<float, GPU_None>::select(THFloatTensor *self, THFloatTensor *src, int dimension, long sliceIndex)
{
    THFloatTensor_select(self, src, dimension, sliceIndex);
}
template<>
void Tensor<double, GPU_None>::select(THDoubleTensor *self, THDoubleTensor *src, int dimension, long sliceIndex)
{
    THDoubleTensor_select(self, src, dimension, sliceIndex);
}

template<>
void Tensor<long, GPU_None>::transpose(THLongTensor *self, THLongTensor *src, int dimension1, int dimension2)
{
    THLongTensor_transpose(self, src, dimension1, dimension2);
}
template<>
void Tensor<float, GPU_None>::transpose(THFloatTensor *self, THFloatTensor *src, int dimension1, int dimension2)
{
    THFloatTensor_transpose(self, src, dimension1, dimension2);
}
template<>
void Tensor<double, GPU_None>::transpose(THDoubleTensor *self, THDoubleTensor *src, int dimension1, int dimension2)
{
    THDoubleTensor_transpose(self, src, dimension1, dimension2);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long, GPU_None>::fill(THLongTensor *r, long val)
{
    return THLongTensor_fill(r, val);
}
template<>
void Tensor<float, GPU_None>::fill(THFloatTensor *r, float val)
{
    return THFloatTensor_fill(r, val);
}
template<>
void Tensor<double, GPU_None>::fill(THDoubleTensor *r, double val)
{
    return THDoubleTensor_fill(r, val);
}

template<>
long Tensor<long, GPU_None>::minall(THLongTensor *r)
{
    return THLongTensor_minall(r);
}
template<>
float Tensor<float, GPU_None>::minall(THFloatTensor *r)
{
    return THFloatTensor_minall(r);
}
template<>
double Tensor<double, GPU_None>::minall(THDoubleTensor *r)
{
    return THDoubleTensor_minall(r);
}

template<>
long Tensor<long, GPU_None>::maxall(THLongTensor *r)
{
    return THLongTensor_maxall(r);
}
template<>
float Tensor<float, GPU_None>::maxall(THFloatTensor *r)
{
    return THFloatTensor_maxall(r);
}
template<>
double Tensor<double, GPU_None>::maxall(THDoubleTensor *r)
{
    return THDoubleTensor_maxall(r);
}

template<>
void Tensor<long, GPU_None>::max(THLongTensor *values, THLongTensor *t, int dimension)
{
    cpptorch::Tensor<long> l(true);
    THLongTensor_max(values, l, t, dimension);
}
template<>
void Tensor<float, GPU_None>::max(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    cpptorch::Tensor<long> l(true);
    THFloatTensor_max(values, l, t, dimension);
}
template<>
void Tensor<double, GPU_None>::max(THDoubleTensor *values, THDoubleTensor *t, int dimension)
{
    cpptorch::Tensor<long> l(true);
    THDoubleTensor_max(values, l, t, dimension);
}

template<>
void Tensor<long, GPU_None>::sum(THLongTensor *values, THLongTensor *t, int dimension)
{
    return THLongTensor_sum(values, t, dimension);
}
template<>
void Tensor<float, GPU_None>::sum(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    return THFloatTensor_sum(values, t, dimension);
}
template<>
void Tensor<double, GPU_None>::sum(THDoubleTensor *values, THDoubleTensor *t, int dimension)
{
    return THDoubleTensor_sum(values, t, dimension);
}

template<>
void Tensor<long, GPU_None>::add(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_add(r, t, val);
}
template<>
void Tensor<float, GPU_None>::add(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_add(r, t, val);
}
template<>
void Tensor<double, GPU_None>::add(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_add(r, t, val);
}

template<>
void Tensor<long, GPU_None>::cadd(THLongTensor *r, THLongTensor *t, long val, THLongTensor *src)
{
    THLongTensor_cadd(r, t, val, src);
}
template<>
void Tensor<float, GPU_None>::cadd(THFloatTensor *r, THFloatTensor *t, float val, THFloatTensor *src)
{
    THFloatTensor_cadd(r, t, val, src);
}
template<>
void Tensor<double, GPU_None>::cadd(THDoubleTensor *r, THDoubleTensor *t, double val, THDoubleTensor *src)
{
    THDoubleTensor_cadd(r, t, val, src);
}

template<>
void Tensor<long, GPU_None>::mul(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_mul(r, t, val);
}
template<>
void Tensor<float, GPU_None>::mul(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_mul(r, t, val);
}
template<>
void Tensor<double, GPU_None>::mul(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_mul(r, t, val);
}

template<>
void Tensor<long, GPU_None>::cmul(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cmul(r, t, src);
}
template<>
void Tensor<float, GPU_None>::cmul(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cmul(r, t, src);
}
template<>
void Tensor<double, GPU_None>::cmul(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cmul(r, t, src);
}

template<>
void Tensor<long, GPU_None>::cdiv(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cdiv(r, t, src);
}
template<>
void Tensor<float, GPU_None>::cdiv(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cdiv(r, t, src);
}
template<>
void Tensor<double, GPU_None>::cdiv(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cdiv(r, t, src);
}

template<>
void Tensor<long, GPU_None>::pow(THLongTensor *r, THLongTensor *t, long val)
{
    //THLongTensor_pow(r, t, val);
    assert(0);
}
template<>
void Tensor<float, GPU_None>::pow(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_pow(r, t, val);
}
template<>
void Tensor<double, GPU_None>::pow(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_pow(r, t, val);
}

template<>
void Tensor<long, GPU_None>::cpow(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cpow(r, t, src);
}
template<>
void Tensor<float, GPU_None>::cpow(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cpow(r, t, src);
}
template<>
void Tensor<double, GPU_None>::cpow(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cpow(r, t, src);
}

template<>
void Tensor<long, GPU_None>::addmv(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat, THLongTensor *vec)
{
    THLongTensor_addmv(r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<float, GPU_None>::addmv(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat, THFloatTensor *vec)
{
    THFloatTensor_addmv(r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<double, GPU_None>::addmv(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat, THDoubleTensor *vec)
{
    THDoubleTensor_addmv(r, beta, t, alpha, mat, vec);
}

template<>
void Tensor<long, GPU_None>::addmm(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat1, THLongTensor *mat2)
{
    THLongTensor_addmm(r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<float, GPU_None>::addmm(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)
{
    THFloatTensor_addmm(r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<double, GPU_None>::addmm(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat1, THDoubleTensor *mat2)
{
    THDoubleTensor_addmm(r, beta, t, alpha, mat1, mat2);
}

template<>
void Tensor<long, GPU_None>::addr(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *vec1, THLongTensor *vec2)
{
    THLongTensor_addr(r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<float, GPU_None>::addr(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *vec1, THFloatTensor *vec2)
{
    THFloatTensor_addr(r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<double, GPU_None>::addr(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *vec1, THDoubleTensor *vec2)
{
    THDoubleTensor_addr(r, beta, t, alpha, vec1, vec2);
}

template<>
void Tensor<long, GPU_None>::abs(THLongTensor *r, THLongTensor *t)
{
    THLongTensor_abs(r, t);
}
template<>
void Tensor<float, GPU_None>::abs(THFloatTensor *r, THFloatTensor *t)
{
    THFloatTensor_abs(r, t);
}
template<>
void Tensor<double, GPU_None>::abs(THDoubleTensor *r, THDoubleTensor *t)
{
    THDoubleTensor_abs(r, t);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
void NN<float, GPU_None>::BatchNormalization_updateOutput(THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *running_mean, THFloatTensor *running_var,
    THFloatTensor *save_mean, THFloatTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_FloatBatchNormalization_updateOutput(nullptr, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}
template<>
void NN<double, GPU_None>::BatchNormalization_updateOutput(THDoubleTensor *input, THDoubleTensor *output,
    THDoubleTensor *weight, THDoubleTensor *bias, THDoubleTensor *running_mean, THDoubleTensor *running_var,
    THDoubleTensor *save_mean, THDoubleTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_DoubleBatchNormalization_updateOutput(nullptr, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}

template<>
void NN<float, GPU_None>::SpatialAveragePooling_updateOutput(THFloatTensor *input, THFloatTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_FloatSpatialAveragePooling_updateOutput(nullptr, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}
template<>
void NN<double, GPU_None>::SpatialAveragePooling_updateOutput(THDoubleTensor *input, THDoubleTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_DoubleSpatialAveragePooling_updateOutput(nullptr, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}

template<>
void NN<float, GPU_None>::SpatialConvolutionMM_updateOutput(THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *finput, THFloatTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_FloatSpatialConvolutionMM_updateOutput(nullptr, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}
template<>
void NN<double, GPU_None>::SpatialConvolutionMM_updateOutput(THDoubleTensor *input, THDoubleTensor *output,
    THDoubleTensor *weight, THDoubleTensor *bias, THDoubleTensor *finput, THDoubleTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_DoubleSpatialConvolutionMM_updateOutput(nullptr, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}

template<>
void NN<float, GPU_None>::SpatialMaxPooling_updateOutput(
    THFloatTensor *input, THFloatTensor *output, THLongTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_FloatSpatialMaxPooling_updateOutput(nullptr, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}
template<>
void NN<double, GPU_None>::SpatialMaxPooling_updateOutput(
    THDoubleTensor *input, THDoubleTensor *output, THLongTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_DoubleSpatialMaxPooling_updateOutput(nullptr, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}

template<>
void NN<float, GPU_None>::SpatialReflectionPadding_updateOutput(
    THFloatTensor *input, THFloatTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_FloatSpatialReflectionPadding_updateOutput(nullptr, input, output, pad_l, pad_r, pad_t, pad_b);
}
template<>
void NN<double, GPU_None>::SpatialReflectionPadding_updateOutput(
    THDoubleTensor *input, THDoubleTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_DoubleSpatialReflectionPadding_updateOutput(nullptr, input, output, pad_l, pad_r, pad_t, pad_b);
}

template<>
void NN<float, GPU_None>::Square_updateOutput(THFloatTensor *input, THFloatTensor *output)
{
    THNN_FloatSquare_updateOutput(nullptr, input, output);
}
template<>
void NN<double, GPU_None>::Square_updateOutput(THDoubleTensor *input, THDoubleTensor *output)
{
    THNN_DoubleSquare_updateOutput(nullptr, input, output);
}

template<>
void NN<float, GPU_None>::Sqrt_updateOutput(THFloatTensor *input, THFloatTensor *output, float eps)
{
    THNN_FloatSqrt_updateOutput(nullptr, input, output, eps);
}
template<>
void NN<double, GPU_None>::Sqrt_updateOutput(THDoubleTensor *input, THDoubleTensor *output, double eps)
{
    THNN_DoubleSqrt_updateOutput(nullptr, input, output, eps);
}

template<>
void NN<float, GPU_None>::Threshold_updateOutput(THFloatTensor *input, THFloatTensor *output,
    float threshold, float val, bool inplace)
{
    THNN_FloatThreshold_updateOutput(nullptr, input, output, threshold, val, inplace);
}
template<>
void NN<double, GPU_None>::Threshold_updateOutput(THDoubleTensor *input, THDoubleTensor *output,
    double threshold, double val, bool inplace)
{
    THNN_DoubleThreshold_updateOutput(nullptr, input, output, threshold, val, inplace);
}

template<>
void NN<float, GPU_None>::SoftMax_updateOutput(THFloatTensor *input, THFloatTensor *output)
{
    THNN_FloatSoftMax_updateOutput(nullptr, input, output);
}
template<>
void NN<double, GPU_None>::SoftMax_updateOutput(THDoubleTensor *input, THDoubleTensor *output)
{
    THNN_DoubleSoftMax_updateOutput(nullptr, input, output);
}

template<>
void NN<float, GPU_None>::LogSoftMax_updateOutput(THFloatTensor *input, THFloatTensor *output)
{
    THNN_FloatLogSoftMax_updateOutput(nullptr, input, output);
}
template<>
void NN<double, GPU_None>::LogSoftMax_updateOutput(THDoubleTensor *input, THDoubleTensor *output)
{
    THNN_DoubleLogSoftMax_updateOutput(nullptr, input, output);
}

}
}
