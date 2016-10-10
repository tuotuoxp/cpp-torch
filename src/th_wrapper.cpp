#include "../include/th_wrapper.h"
#include "../include/torch/Storage.h"
#include "../include/torch/Tensor.h"

#include <TH/TH.h>
#include <THNN/THNN.h>
#include <assert.h>


template<>
THFloatStorage* cpptorch::th::Storage<StorageFloat>::newWithDataAndAllocator(float *data, long size, THAllocator *allocator, void *allocatorContext)
{
    return THFloatStorage_newWithDataAndAllocator(data, size, allocator, allocatorContext);
}

template<>
THLongStorage* cpptorch::th::Storage<StorageLong>::newWithDataAndAllocator(long *data, long size, THAllocator *allocator, void *allocatorContext)
{
    return THLongStorage_newWithDataAndAllocator(data, size, allocator, allocatorContext);
}

template<>
void cpptorch::th::Storage<StorageFloat>::retain(THFloatStorage *storage)
{
    THFloatStorage_retain(storage);
}

template<>
void cpptorch::th::Storage<StorageLong>::retain(THLongStorage *storage)
{
    THLongStorage_retain(storage);
}

template<>
void cpptorch::th::Storage<StorageFloat>::release(THFloatStorage *storage)
{
    THFloatStorage_free(storage);
}

template<>
void cpptorch::th::Storage<StorageLong>::release(THLongStorage *storage)
{
    THLongStorage_free(storage);
}

//////////////////////////////////////////////////////////////////////////

template<>
float* cpptorch::th::Storage<StorageFloat>::data(THFloatStorage *storage)
{
    return THFloatStorage_data(storage);
}

template<>
long* cpptorch::th::Storage<StorageLong>::data(THLongStorage *storage)
{
    return THLongStorage_data(storage);
}

template<>
int cpptorch::th::Storage<StorageFloat>::size(THFloatStorage *storage)
{
    return THFloatStorage_size(storage);
}

template<>
int cpptorch::th::Storage<StorageLong>::size(THLongStorage *storage)
{
    return THLongStorage_size(storage);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
THLongTensor* cpptorch::th::Tensor<TensorLong>::create()
{
    return THLongTensor_new();
}
template<>
THFloatTensor* cpptorch::th::Tensor<TensorFloat>::create()
{
    return THFloatTensor_new();
}

template<>
THLongTensor* cpptorch::th::Tensor<TensorLong>::newWithStorage(THLongStorage *storage, long offset, THLongStorage *size, THLongStorage *stride)
{
    return THLongTensor_newWithStorage(storage, offset, size, stride);
}
template<>
THFloatTensor* cpptorch::th::Tensor<TensorFloat>::newWithStorage(THFloatStorage *storage, long offset, THLongStorage *size, THLongStorage *stride)
{
    return THFloatTensor_newWithStorage(storage, offset, size, stride);
}

template<>
void cpptorch::th::Tensor<TensorLong>::resize(THLongTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THLongTensor_resize(self, size, stride);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::resize(THFloatTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THFloatTensor_resize(self, size, stride);
}

template<>
void cpptorch::th::Tensor<TensorLong>::resizeAs(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_resizeAs(self, src);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::resizeAs(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_resizeAs(self, src);
}

template<>
void cpptorch::th::Tensor<TensorLong>::copy(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_copy(self, src);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::copy(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_copy(self, src);
}

template<>
void cpptorch::th::Tensor<TensorLong>::retain(THLongTensor *tensor)
{
    THLongTensor_retain(tensor);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::retain(THFloatTensor *tensor)
{
    THFloatTensor_retain(tensor);
}

template<>
void cpptorch::th::Tensor<TensorLong>::release(THLongTensor *tensor)
{
    THLongTensor_free(tensor);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::release(THFloatTensor *tensor)
{
    THFloatTensor_free(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
THLongStorage* cpptorch::th::Tensor<TensorLong>::storage(const THLongTensor *tensor)
{
    return THLongTensor_storage(tensor);
}
template<>
THFloatStorage* cpptorch::th::Tensor<TensorFloat>::storage(const THFloatTensor *tensor)
{
    return THFloatTensor_storage(tensor);
}

template<>
long cpptorch::th::Tensor<TensorLong>::storageOffset(const THLongTensor *tensor)
{
    return THLongTensor_storageOffset(tensor);
}
template<>
long cpptorch::th::Tensor<TensorFloat>::storageOffset(const THFloatTensor *tensor)
{
    return THFloatTensor_storageOffset(tensor);
}

template<>
int cpptorch::th::Tensor<TensorLong>::nDimension(const THLongTensor *tensor)
{
    return THLongTensor_nDimension(tensor);
}
template<>
int cpptorch::th::Tensor<TensorFloat>::nDimension(const THFloatTensor *tensor)
{
    return THFloatTensor_nDimension(tensor);
}

template<>
THLongStorage* cpptorch::th::Tensor<TensorLong>::size(const THLongTensor *tensor)
{
    return THLongTensor_newSizeOf((THLongTensor*)tensor);
}
template<>
THLongStorage* cpptorch::th::Tensor<TensorFloat>::size(const THFloatTensor *tensor)
{
    return THFloatTensor_newSizeOf((THFloatTensor*)tensor);
}

template<>
long cpptorch::th::Tensor<TensorLong>::size(const THLongTensor *tensor, int dim)
{
    return THLongTensor_size(tensor, dim);
}
template<>
long cpptorch::th::Tensor<TensorFloat>::size(const THFloatTensor *tensor, int dim)
{
    return THFloatTensor_size(tensor, dim);
}

template<>
THLongStorage* cpptorch::th::Tensor<TensorLong>::stride(const THLongTensor *tensor)
{
    return THLongTensor_newStrideOf((THLongTensor*)tensor);
}
template<>
THLongStorage* cpptorch::th::Tensor<TensorFloat>::stride(const THFloatTensor *tensor)
{
    return THFloatTensor_newStrideOf((THFloatTensor*)tensor);
}

template<>
long *cpptorch::th::Tensor<TensorLong>::data(const THLongTensor *tensor)
{
    return THLongTensor_data((THLongTensor*)tensor);
}
template<>
float *cpptorch::th::Tensor<TensorFloat>::data(const THFloatTensor *tensor)
{
    return THFloatTensor_data((THFloatTensor*)tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
int cpptorch::th::Tensor<TensorLong>::isContiguous(const THLongTensor *tensor)
{
    return THLongTensor_isContiguous(tensor);
}
template<>
int cpptorch::th::Tensor<TensorFloat>::isContiguous(const THFloatTensor *tensor)
{
    return THFloatTensor_isContiguous(tensor);
}

template<>
int cpptorch::th::Tensor<TensorLong>::nElement(const THLongTensor *tensor)
{
    return THLongTensor_nElement(tensor);
}
template<>
int cpptorch::th::Tensor<TensorFloat>::nElement(const THFloatTensor *tensor)
{
    return THFloatTensor_nElement(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
void cpptorch::th::Tensor<TensorLong>::narrow(THLongTensor *self, THLongTensor *src, int dimension, long firstIndex, long size)
{
    THLongTensor_narrow(self, src, dimension, firstIndex, size);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::narrow(THFloatTensor *self, THFloatTensor *src, int dimension, long firstIndex, long size)
{
    THFloatTensor_narrow(self, src, dimension, firstIndex, size);
}

template<>
void cpptorch::th::Tensor<TensorLong>::select(THLongTensor *self, THLongTensor *src, int dimension, long sliceIndex)
{
    THLongTensor_select(self, src, dimension, sliceIndex);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::select(THFloatTensor *self, THFloatTensor *src, int dimension, long sliceIndex)
{
    THFloatTensor_select(self, src, dimension, sliceIndex);
}

template<>
void cpptorch::th::Tensor<TensorLong>::transpose(THLongTensor *self, THLongTensor *src, int dimension1, int dimension2)
{
    THLongTensor_transpose(self, src, dimension1, dimension2);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::transpose(THFloatTensor *self, THFloatTensor *src, int dimension1, int dimension2)
{
    THFloatTensor_transpose(self, src, dimension1, dimension2);
}

//////////////////////////////////////////////////////////////////////////

template<>
void cpptorch::th::Tensor<TensorLong>::fill(THLongTensor *r, long val)
{
    return THLongTensor_fill(r, val);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::fill(THFloatTensor *r, float val)
{
    return THFloatTensor_fill(r, val);
}

template<>
long cpptorch::th::Tensor<TensorLong>::minall(THLongTensor *r)
{
    return THLongTensor_minall(r);
}
template<>
float cpptorch::th::Tensor<TensorFloat>::minall(THFloatTensor *r)
{
    return THFloatTensor_minall(r);
}

template<>
long cpptorch::th::Tensor<TensorLong>::maxall(THLongTensor *r)
{
    return THLongTensor_maxall(r);
}
template<>
float cpptorch::th::Tensor<TensorFloat>::maxall(THFloatTensor *r)
{
    return THFloatTensor_maxall(r);
}

template<>
void cpptorch::th::Tensor<TensorLong>::max(THLongTensor *values, THLongTensor *t, int dimension)
{
    THLongTensor *l = THLongTensor_new();
    THLongTensor_max(values, l, t, dimension);
    THLongTensor_free(l);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::max(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    THLongTensor *l = THLongTensor_new();
    THFloatTensor_max(values, l, t, dimension);
    THLongTensor_free(l);
}

template<>
void cpptorch::th::Tensor<TensorLong>::sum(THLongTensor *values, THLongTensor *t, int dimension)
{
    return THLongTensor_sum(values, t, dimension);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::sum(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    return THFloatTensor_sum(values, t, dimension);
}

template<>
void cpptorch::th::Tensor<TensorLong>::add(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_add(r, t, val);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::add(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_add(r, t, val);
}

template<>
void cpptorch::th::Tensor<TensorLong>::cadd(THLongTensor *r, THLongTensor *t, long val, THLongTensor *src)
{
    THLongTensor_cadd(r, t, val, src);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::cadd(THFloatTensor *r, THFloatTensor *t, float val, THFloatTensor *src)
{
    THFloatTensor_cadd(r, t, val, src);
}

template<>
void cpptorch::th::Tensor<TensorLong>::mul(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_mul(r, t, val);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::mul(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_mul(r, t, val);
}

template<>
void cpptorch::th::Tensor<TensorLong>::cmul(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cmul(r, t, src);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::cmul(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cmul(r, t, src);
}

template<>
void cpptorch::th::Tensor<TensorLong>::cdiv(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cdiv(r, t, src);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::cdiv(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cdiv(r, t, src);
}

template<>
void cpptorch::th::Tensor<TensorLong>::pow(THLongTensor *r, THLongTensor *t, long val)
{
    //THLongTensor_pow(r, t, val);
    assert(0);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::pow(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_pow(r, t, val);
}

template<>
void cpptorch::th::Tensor<TensorLong>::cpow(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cpow(r, t, src);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::cpow(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cpow(r, t, src);
}

template<>
void cpptorch::th::Tensor<TensorLong>::addmv(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat, THLongTensor *vec)
{
    THLongTensor_addmv(r, beta, t, alpha, mat, vec);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::addmv(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat, THFloatTensor *vec)
{
    THFloatTensor_addmv(r, beta, t, alpha, mat, vec);
}

template<>
void cpptorch::th::Tensor<TensorLong>::addmm(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat1, THLongTensor *mat2)
{
    THLongTensor_addmm(r, beta, t, alpha, mat1, mat2);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::addmm(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)
{
    THFloatTensor_addmm(r, beta, t, alpha, mat1, mat2);
}

template<>
void cpptorch::th::Tensor<TensorLong>::addr(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *vec1, THLongTensor *vec2)
{
    THLongTensor_addr(r, beta, t, alpha, vec1, vec2);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::addr(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *vec1, THFloatTensor *vec2)
{
    THFloatTensor_addr(r, beta, t, alpha, vec1, vec2);
}

template<>
void cpptorch::th::Tensor<TensorLong>::abs(THLongTensor *r, THLongTensor *t)
{
    THLongTensor_abs(r, t);
}
template<>
void cpptorch::th::Tensor<TensorFloat>::abs(THFloatTensor *r, THFloatTensor *t)
{
    THFloatTensor_abs(r, t);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<>
void cpptorch::th::NN<TensorFloat>::BatchNormalization_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *running_mean, THFloatTensor *running_var,
    THFloatTensor *save_mean, THFloatTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_FloatBatchNormalization_updateOutput(state, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}

template<>
void cpptorch::th::NN<TensorFloat>::SpatialAveragePooling_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_FloatSpatialAveragePooling_updateOutput(state, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}

template<>
void cpptorch::th::NN<TensorFloat>::SpatialConvolutionMM_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *finput, THFloatTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_FloatSpatialConvolutionMM_updateOutput(state, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}

template<>
void cpptorch::th::NN<TensorFloat>::SpatialMaxPooling_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output, THFloatTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_FloatSpatialMaxPooling_updateOutput(state, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}

template<>
void cpptorch::th::NN<TensorFloat>::SpatialReflectionPadding_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_FloatSpatialReflectionPadding_updateOutput(state, input, output, pad_l, pad_r, pad_t, pad_b);
}

template<>
void cpptorch::th::NN<TensorFloat>::Square_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output)
{
    THNN_FloatSquare_updateOutput(state, input, output);
}

template<>
void cpptorch::th::NN<TensorFloat>::Sqrt_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output, float eps)
{
    THNN_FloatSqrt_updateOutput(state, input, output, eps);
}

template<>
void cpptorch::th::NN<TensorFloat>::Threshold_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    float threshold, float val, bool inplace)
{
    THNN_FloatThreshold_updateOutput(state, input, output, threshold, val, inplace);
}
