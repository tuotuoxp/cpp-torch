#include "th_wrapper.h"
#include "../include/torch/Storage.h"
#include "../include/torch/Tensor.h"
#include "allocator.h"

#include <TH/TH.h>
#include <THNN/THNN.h>
#include <assert.h>


template<typename T>
static T* bypass(const T *ptr_src, long size, bool take_ownership_of_data)
{
    if (!take_ownership_of_data)
    {
        long sz = size * sizeof(T);
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
THLongStorage* Storage<long, false>::newWithData(const long *ptr_src, long size, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        return THLongStorage_newWithDataAndAllocator(bypass(ptr_src, size, take_ownership_of_data), size,
            cpptorch::allocator::get(), cpptorch::allocator::requestIndex(size));
    }
    else
    {
        return THLongStorage_newWithAllocator(0, cpptorch::allocator::get(), cpptorch::allocator::requestIndex(0));
    }
}
template<>
THFloatStorage* Storage<float, false>::newWithData(const float *ptr_src, long size, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        return THFloatStorage_newWithDataAndAllocator(bypass(ptr_src, size, take_ownership_of_data), size,
            cpptorch::allocator::get(), cpptorch::allocator::requestIndex(size));
    }
    else
    {
        return THFloatStorage_newWithAllocator(0, cpptorch::allocator::get(), cpptorch::allocator::requestIndex(0));
    }
}
template<>
THDoubleStorage* Storage<double, false>::newWithData(const double *ptr_src, long size, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        return THDoubleStorage_newWithDataAndAllocator(bypass(ptr_src, size, take_ownership_of_data), size,
            cpptorch::allocator::get(), cpptorch::allocator::requestIndex(size));
    }
    else
    {
        return THDoubleStorage_newWithAllocator(0, cpptorch::allocator::get(), cpptorch::allocator::requestIndex(0));
    }
}

template<>
void Storage<long, false>::retain(THLongStorage *storage)
{
    THLongStorage_retain(storage);
}
template<>
void Storage<float, false>::retain(THFloatStorage *storage)
{
    THFloatStorage_retain(storage);
}
template<>
void Storage<double, false>::retain(THDoubleStorage *storage)
{
    THDoubleStorage_retain(storage);
}

template<>
void Storage<long, false>::release(THLongStorage *storage)
{
    THLongStorage_free(storage);
}
template<>
void Storage<float, false>::release(THFloatStorage *storage)
{
    THFloatStorage_free(storage);
}
template<>
void Storage<double, false>::release(THDoubleStorage *storage)
{
    THDoubleStorage_free(storage);
}

//////////////////////////////////////////////////////////////////////////

template<>
long* Storage<long, false>::data(THLongStorage *storage)
{
    return THLongStorage_data(storage);
}
template<>
float* Storage<float, false>::data(THFloatStorage *storage)
{
    return THFloatStorage_data(storage);
}
template<>
double* Storage<double, false>::data(THDoubleStorage *storage)
{
    return THDoubleStorage_data(storage);
}

template<>
long Storage<long, false>::size(THLongStorage *storage)
{
    return (long)THLongStorage_size(storage);
}
template<>
long Storage<float, false>::size(THFloatStorage *storage)
{
    return (long)THFloatStorage_size(storage);
}
template<>
long Storage<double, false>::size(THDoubleStorage *storage)
{
    return (long)THDoubleStorage_size(storage);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
THLongTensor* Tensor<long, false>::newWithStorage(THLongStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THLongTensor_newWithStorage(storage, offset, nullptr, nullptr);
    case 1: return THLongTensor_newWithStorage1d(storage, offset, size[0], stride[0]);
    case 2: return THLongTensor_newWithStorage2d(storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THLongTensor_newWithStorage3d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THLongTensor_newWithStorage4d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THLongTensor_newWithStorage(storage, offset, cpptorch::Storage<long>(size, dim, false), cpptorch::Storage<long>(stride, dim, false));
}
template<>
THFloatTensor* Tensor<float, false>::newWithStorage(THFloatStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THFloatTensor_newWithStorage(storage, offset, nullptr, nullptr);
    case 1: return THFloatTensor_newWithStorage1d(storage, offset, size[0], stride[0]);
    case 2: return THFloatTensor_newWithStorage2d(storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THFloatTensor_newWithStorage3d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THFloatTensor_newWithStorage4d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THFloatTensor_newWithStorage(storage, offset, cpptorch::Storage<long>(size, dim, false), cpptorch::Storage<long>(stride, dim, false));
}
template<>
THDoubleTensor* Tensor<double, false>::newWithStorage(THDoubleStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THDoubleTensor_newWithStorage(storage, offset, nullptr, nullptr);
    case 1: return THDoubleTensor_newWithStorage1d(storage, offset, size[0], stride[0]);
    case 2: return THDoubleTensor_newWithStorage2d(storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THDoubleTensor_newWithStorage3d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THDoubleTensor_newWithStorage4d(storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THDoubleTensor_newWithStorage(storage, offset, cpptorch::Storage<long>(size, dim, false), cpptorch::Storage<long>(stride, dim, false));
}

template<>
void Tensor<long, false>::resize(THLongTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THLongTensor_resize(self, size, stride);
}
template<>
void Tensor<float, false>::resize(THFloatTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THFloatTensor_resize(self, size, stride);
}
template<>
void Tensor<double, false>::resize(THDoubleTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THDoubleTensor_resize(self, size, stride);
}

template<>
void Tensor<long, false>::resizeAs(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_resizeAs(self, src);
}
template<>
void Tensor<float, false>::resizeAs(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_resizeAs(self, src);
}
template<>
void Tensor<double, false>::resizeAs(THDoubleTensor *self, THDoubleTensor *src)
{
    THDoubleTensor_resizeAs(self, src);
}

template<>
void Tensor<long, false>::copy(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_copy(self, src);
}
template<>
void Tensor<float, false>::copy(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_copy(self, src);
}
template<>
void Tensor<double, false>::copy(THDoubleTensor *self, THDoubleTensor *src)
{
    THDoubleTensor_copy(self, src);
}

template<>
void Tensor<long, false>::retain(THLongTensor *tensor)
{
    THLongTensor_retain(tensor);
}
template<>
void Tensor<float, false>::retain(THFloatTensor *tensor)
{
    THFloatTensor_retain(tensor);
}
template<>
void Tensor<double, false>::retain(THDoubleTensor *tensor)
{
    THDoubleTensor_retain(tensor);
}

template<>
void Tensor<long, false>::release(THLongTensor *tensor)
{
    THLongTensor_free(tensor);
}
template<>
void Tensor<float, false>::release(THFloatTensor *tensor)
{
    THFloatTensor_free(tensor);
}
template<>
void Tensor<double, false>::release(THDoubleTensor *tensor)
{
    THDoubleTensor_free(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
THLongStorage* Tensor<long, false>::storage(const THLongTensor *tensor)
{
    return THLongTensor_storage(tensor);
}
template<>
THFloatStorage* Tensor<float, false>::storage(const THFloatTensor *tensor)
{
    return THFloatTensor_storage(tensor);
}
template<>
THDoubleStorage* Tensor<double, false>::storage(const THDoubleTensor *tensor)
{
    return THDoubleTensor_storage(tensor);
}

template<>
long Tensor<long, false>::storageOffset(const THLongTensor *tensor)
{
    return (long)THLongTensor_storageOffset(tensor);
}
template<>
long Tensor<float, false>::storageOffset(const THFloatTensor *tensor)
{
    return (long)THFloatTensor_storageOffset(tensor);
}
template<>
long Tensor<double, false>::storageOffset(const THDoubleTensor *tensor)
{
    return (long)THDoubleTensor_storageOffset(tensor);
}

template<>
int Tensor<long, false>::nDimension(const THLongTensor *tensor)
{
    return THLongTensor_nDimension(tensor);
}
template<>
int Tensor<float, false>::nDimension(const THFloatTensor *tensor)
{
    return THFloatTensor_nDimension(tensor);
}
template<>
int Tensor<double, false>::nDimension(const THDoubleTensor *tensor)
{
    return THDoubleTensor_nDimension(tensor);
}

template<>
THLongStorage* Tensor<long, false>::size(const THLongTensor *tensor)
{
    return THLongTensor_newSizeOf((THLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float, false>::size(const THFloatTensor *tensor)
{
    return THFloatTensor_newSizeOf((THFloatTensor*)tensor);
}
template<>
THLongStorage* Tensor<double, false>::size(const THDoubleTensor *tensor)
{
    return THDoubleTensor_newSizeOf((THDoubleTensor*)tensor);
}

template<>
long Tensor<long, false>::size(const THLongTensor *tensor, int dim)
{
    return THLongTensor_size(tensor, dim);
}
template<>
long Tensor<float, false>::size(const THFloatTensor *tensor, int dim)
{
    return THFloatTensor_size(tensor, dim);
}
template<>
long Tensor<double, false>::size(const THDoubleTensor *tensor, int dim)
{
    return THDoubleTensor_size(tensor, dim);
}

template<>
THLongStorage* Tensor<long, false>::stride(const THLongTensor *tensor)
{
    return THLongTensor_newStrideOf((THLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float, false>::stride(const THFloatTensor *tensor)
{
    return THFloatTensor_newStrideOf((THFloatTensor*)tensor);
}
template<>
THLongStorage* Tensor<double, false>::stride(const THDoubleTensor *tensor)
{
    return THDoubleTensor_newStrideOf((THDoubleTensor*)tensor);
}

template<>
long *Tensor<long, false>::data(const THLongTensor *tensor)
{
    return THLongTensor_data((THLongTensor*)tensor);
}
template<>
float *Tensor<float, false>::data(const THFloatTensor *tensor)
{
    return THFloatTensor_data((THFloatTensor*)tensor);
}
template<>
double *Tensor<double, false>::data(const THDoubleTensor *tensor)
{
    return THDoubleTensor_data((THDoubleTensor*)tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
int Tensor<long, false>::isContiguous(const THLongTensor *tensor)
{
    return THLongTensor_isContiguous(tensor);
}
template<>
int Tensor<float, false>::isContiguous(const THFloatTensor *tensor)
{
    return THFloatTensor_isContiguous(tensor);
}
template<>
int Tensor<double, false>::isContiguous(const THDoubleTensor *tensor)
{
    return THDoubleTensor_isContiguous(tensor);
}

template<>
long Tensor<long, false>::nElement(const THLongTensor *tensor)
{
    return (long)THLongTensor_nElement(tensor);
}
template<>
long Tensor<float, false>::nElement(const THFloatTensor *tensor)
{
    return (long)THFloatTensor_nElement(tensor);
}
template<>
long Tensor<double, false>::nElement(const THDoubleTensor *tensor)
{
    return (long)THDoubleTensor_nElement(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long, false>::narrow(THLongTensor *self, THLongTensor *src, int dimension, long firstIndex, long size)
{
    THLongTensor_narrow(self, src, dimension, firstIndex, size);
}
template<>
void Tensor<float, false>::narrow(THFloatTensor *self, THFloatTensor *src, int dimension, long firstIndex, long size)
{
    THFloatTensor_narrow(self, src, dimension, firstIndex, size);
}
template<>
void Tensor<double, false>::narrow(THDoubleTensor *self, THDoubleTensor *src, int dimension, long firstIndex, long size)
{
    THDoubleTensor_narrow(self, src, dimension, firstIndex, size);
}

template<>
void Tensor<long, false>::select(THLongTensor *self, THLongTensor *src, int dimension, long sliceIndex)
{
    THLongTensor_select(self, src, dimension, sliceIndex);
}
template<>
void Tensor<float, false>::select(THFloatTensor *self, THFloatTensor *src, int dimension, long sliceIndex)
{
    THFloatTensor_select(self, src, dimension, sliceIndex);
}
template<>
void Tensor<double, false>::select(THDoubleTensor *self, THDoubleTensor *src, int dimension, long sliceIndex)
{
    THDoubleTensor_select(self, src, dimension, sliceIndex);
}

template<>
void Tensor<long, false>::transpose(THLongTensor *self, THLongTensor *src, int dimension1, int dimension2)
{
    THLongTensor_transpose(self, src, dimension1, dimension2);
}
template<>
void Tensor<float, false>::transpose(THFloatTensor *self, THFloatTensor *src, int dimension1, int dimension2)
{
    THFloatTensor_transpose(self, src, dimension1, dimension2);
}
template<>
void Tensor<double, false>::transpose(THDoubleTensor *self, THDoubleTensor *src, int dimension1, int dimension2)
{
    THDoubleTensor_transpose(self, src, dimension1, dimension2);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long, false>::fill(THLongTensor *r, long val)
{
    return THLongTensor_fill(r, val);
}
template<>
void Tensor<float, false>::fill(THFloatTensor *r, float val)
{
    return THFloatTensor_fill(r, val);
}
template<>
void Tensor<double, false>::fill(THDoubleTensor *r, double val)
{
    return THDoubleTensor_fill(r, val);
}

template<>
long Tensor<long, false>::minall(THLongTensor *r)
{
    return THLongTensor_minall(r);
}
template<>
float Tensor<float, false>::minall(THFloatTensor *r)
{
    return THFloatTensor_minall(r);
}
template<>
double Tensor<double, false>::minall(THDoubleTensor *r)
{
    return THDoubleTensor_minall(r);
}

template<>
long Tensor<long, false>::maxall(THLongTensor *r)
{
    return THLongTensor_maxall(r);
}
template<>
float Tensor<float, false>::maxall(THFloatTensor *r)
{
    return THFloatTensor_maxall(r);
}
template<>
double Tensor<double, false>::maxall(THDoubleTensor *r)
{
    return THDoubleTensor_maxall(r);
}

template<>
void Tensor<long, false>::max(THLongTensor *values, THLongTensor *t, int dimension)
{
    cpptorch::Tensor<long> l(true);
    THLongTensor_max(values, l, t, dimension);
}
template<>
void Tensor<float, false>::max(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    cpptorch::Tensor<long> l(true);
    THFloatTensor_max(values, l, t, dimension);
}
template<>
void Tensor<double, false>::max(THDoubleTensor *values, THDoubleTensor *t, int dimension)
{
    cpptorch::Tensor<long> l(true);
    THDoubleTensor_max(values, l, t, dimension);
}

template<>
void Tensor<long, false>::sum(THLongTensor *values, THLongTensor *t, int dimension)
{
    return THLongTensor_sum(values, t, dimension);
}
template<>
void Tensor<float, false>::sum(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    return THFloatTensor_sum(values, t, dimension);
}
template<>
void Tensor<double, false>::sum(THDoubleTensor *values, THDoubleTensor *t, int dimension)
{
    return THDoubleTensor_sum(values, t, dimension);
}

template<>
void Tensor<long, false>::add(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_add(r, t, val);
}
template<>
void Tensor<float, false>::add(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_add(r, t, val);
}
template<>
void Tensor<double, false>::add(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_add(r, t, val);
}

template<>
void Tensor<long, false>::cadd(THLongTensor *r, THLongTensor *t, long val, THLongTensor *src)
{
    THLongTensor_cadd(r, t, val, src);
}
template<>
void Tensor<float, false>::cadd(THFloatTensor *r, THFloatTensor *t, float val, THFloatTensor *src)
{
    THFloatTensor_cadd(r, t, val, src);
}
template<>
void Tensor<double, false>::cadd(THDoubleTensor *r, THDoubleTensor *t, double val, THDoubleTensor *src)
{
    THDoubleTensor_cadd(r, t, val, src);
}

template<>
void Tensor<long, false>::mul(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_mul(r, t, val);
}
template<>
void Tensor<float, false>::mul(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_mul(r, t, val);
}
template<>
void Tensor<double, false>::mul(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_mul(r, t, val);
}

template<>
void Tensor<long, false>::cmul(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cmul(r, t, src);
}
template<>
void Tensor<float, false>::cmul(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cmul(r, t, src);
}
template<>
void Tensor<double, false>::cmul(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cmul(r, t, src);
}

template<>
void Tensor<long, false>::cdiv(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cdiv(r, t, src);
}
template<>
void Tensor<float, false>::cdiv(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cdiv(r, t, src);
}
template<>
void Tensor<double, false>::cdiv(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cdiv(r, t, src);
}

template<>
void Tensor<long, false>::pow(THLongTensor *r, THLongTensor *t, long val)
{
    //THLongTensor_pow(r, t, val);
    assert(0);
}
template<>
void Tensor<float, false>::pow(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_pow(r, t, val);
}
template<>
void Tensor<double, false>::pow(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_pow(r, t, val);
}

template<>
void Tensor<long, false>::cpow(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cpow(r, t, src);
}
template<>
void Tensor<float, false>::cpow(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cpow(r, t, src);
}
template<>
void Tensor<double, false>::cpow(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cpow(r, t, src);
}

template<>
void Tensor<long, false>::addmv(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat, THLongTensor *vec)
{
    THLongTensor_addmv(r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<float, false>::addmv(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat, THFloatTensor *vec)
{
    THFloatTensor_addmv(r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<double, false>::addmv(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat, THDoubleTensor *vec)
{
    THDoubleTensor_addmv(r, beta, t, alpha, mat, vec);
}

template<>
void Tensor<long, false>::addmm(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat1, THLongTensor *mat2)
{
    THLongTensor_addmm(r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<float, false>::addmm(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)
{
    THFloatTensor_addmm(r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<double, false>::addmm(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat1, THDoubleTensor *mat2)
{
    THDoubleTensor_addmm(r, beta, t, alpha, mat1, mat2);
}

template<>
void Tensor<long, false>::addr(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *vec1, THLongTensor *vec2)
{
    THLongTensor_addr(r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<float, false>::addr(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *vec1, THFloatTensor *vec2)
{
    THFloatTensor_addr(r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<double, false>::addr(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *vec1, THDoubleTensor *vec2)
{
    THDoubleTensor_addr(r, beta, t, alpha, vec1, vec2);
}

template<>
void Tensor<long, false>::abs(THLongTensor *r, THLongTensor *t)
{
    THLongTensor_abs(r, t);
}
template<>
void Tensor<float, false>::abs(THFloatTensor *r, THFloatTensor *t)
{
    THFloatTensor_abs(r, t);
}
template<>
void Tensor<double, false>::abs(THDoubleTensor *r, THDoubleTensor *t)
{
    THDoubleTensor_abs(r, t);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
void NN<float, false>::BatchNormalization_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *running_mean, THFloatTensor *running_var,
    THFloatTensor *save_mean, THFloatTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_FloatBatchNormalization_updateOutput(state, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}
template<>
void NN<double, false>::BatchNormalization_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    THDoubleTensor *weight, THDoubleTensor *bias, THDoubleTensor *running_mean, THDoubleTensor *running_var,
    THDoubleTensor *save_mean, THDoubleTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_DoubleBatchNormalization_updateOutput(state, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}

template<>
void NN<float, false>::SpatialAveragePooling_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_FloatSpatialAveragePooling_updateOutput(state, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}
template<>
void NN<double, false>::SpatialAveragePooling_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_DoubleSpatialAveragePooling_updateOutput(state, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}

template<>
void NN<float, false>::SpatialConvolutionMM_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *finput, THFloatTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_FloatSpatialConvolutionMM_updateOutput(state, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}
template<>
void NN<double, false>::SpatialConvolutionMM_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    THDoubleTensor *weight, THDoubleTensor *bias, THDoubleTensor *finput, THDoubleTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_DoubleSpatialConvolutionMM_updateOutput(state, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}

template<>
void NN<float, false>::SpatialMaxPooling_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output, THFloatTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_FloatSpatialMaxPooling_updateOutput(state, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}
template<>
void NN<double, false>::SpatialMaxPooling_updateOutput(THNNState *state,
    THDoubleTensor *input, THDoubleTensor *output, THDoubleTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_DoubleSpatialMaxPooling_updateOutput(state, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}

template<>
void NN<float, false>::SpatialReflectionPadding_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_FloatSpatialReflectionPadding_updateOutput(state, input, output, pad_l, pad_r, pad_t, pad_b);
}
template<>
void NN<double, false>::SpatialReflectionPadding_updateOutput(THNNState *state,
    THDoubleTensor *input, THDoubleTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_DoubleSpatialReflectionPadding_updateOutput(state, input, output, pad_l, pad_r, pad_t, pad_b);
}

template<>
void NN<float, false>::Square_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output)
{
    THNN_FloatSquare_updateOutput(state, input, output);
}
template<>
void NN<double, false>::Square_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output)
{
    THNN_DoubleSquare_updateOutput(state, input, output);
}

template<>
void NN<float, false>::Sqrt_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output, float eps)
{
    THNN_FloatSqrt_updateOutput(state, input, output, eps);
}
template<>
void NN<double, false>::Sqrt_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output, double eps)
{
    THNN_DoubleSqrt_updateOutput(state, input, output, eps);
}

template<>
void NN<float, false>::Threshold_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    float threshold, float val, bool inplace)
{
    THNN_FloatThreshold_updateOutput(state, input, output, threshold, val, inplace);
}
template<>
void NN<double, false>::Threshold_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    double threshold, double val, bool inplace)
{
    THNN_DoubleThreshold_updateOutput(state, input, output, threshold, val, inplace);
}


}
}
