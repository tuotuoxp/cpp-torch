#include "th_wrapper.h"
#include "nn/Storage.h"
#include "nn/Tensor.h"


template<>
THFloatStorage* THWrapper::Storage<StorageFloat>::newWithDataAndAllocator(float *data, long size, THAllocator *allocator, void *allocatorContext)
{
    return THFloatStorage_newWithDataAndAllocator(data, size, allocator, allocatorContext);
}

template<>
THLongStorage* THWrapper::Storage<StorageLong>::newWithDataAndAllocator(long *data, long size, THAllocator *allocator, void *allocatorContext)
{
    return THLongStorage_newWithDataAndAllocator(data, size, allocator, allocatorContext);
}

template<>
void THWrapper::Storage<StorageFloat>::retain(THFloatStorage *storage)
{
    THFloatStorage_retain(storage);
}

template<>
void THWrapper::Storage<StorageLong>::retain(THLongStorage *storage)
{
    THLongStorage_retain(storage);
}

template<>
void THWrapper::Storage<StorageFloat>::release(THFloatStorage *storage)
{
    THFloatStorage_free(storage);
}

template<>
void THWrapper::Storage<StorageLong>::release(THLongStorage *storage)
{
    THLongStorage_free(storage);
}

//////////////////////////////////////////////////////////////////////////

template<>
float* THWrapper::Storage<StorageFloat>::data(THFloatStorage *storage)
{
    return THFloatStorage_data(storage);
}

template<>
long* THWrapper::Storage<StorageLong>::data(THLongStorage *storage)
{
    return THLongStorage_data(storage);
}

template<>
int THWrapper::Storage<StorageFloat>::size(THFloatStorage *storage)
{
    return THFloatStorage_size(storage);
}

template<>
int THWrapper::Storage<StorageLong>::size(THLongStorage *storage)
{
    return THLongStorage_size(storage);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
THFloatTensor* THWrapper::Tensor<TensorFloat>::create()
{
    return THFloatTensor_new();
}

template<>
THFloatTensor* THWrapper::Tensor<TensorFloat>::newWithStorage(THFloatStorage *storage, long offset, THLongStorage *size, THLongStorage *stride)
{
    return THFloatTensor_newWithStorage(storage, offset, size, stride);
}

template<>
void THWrapper::Tensor<TensorFloat>::resize(THFloatTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THFloatTensor_resize(self, size, stride);
}

template<>
void THWrapper::Tensor<TensorFloat>::resizeAs(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_resizeAs(self, src);
}

template<>
void THWrapper::Tensor<TensorFloat>::copy(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_copy(self, src);
}

template<>
void THWrapper::Tensor<TensorFloat>::retain(THFloatTensor *tensor)
{
    THFloatTensor_retain(tensor);
}

template<>
void THWrapper::Tensor<TensorFloat>::release(THFloatTensor *tensor)
{
    THFloatTensor_free(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
THFloatStorage* THWrapper::Tensor<TensorFloat>::storage(const THFloatTensor *tensor)
{
    return THFloatTensor_storage(tensor);
}

template<>
long THWrapper::Tensor<TensorFloat>::storageOffset(const THFloatTensor *tensor)
{
    return THFloatTensor_storageOffset(tensor);
}

template<>
int THWrapper::Tensor<TensorFloat>::nDimension(const THFloatTensor *tensor)
{
    return THFloatTensor_nDimension(tensor);
}

template<>
THLongStorage* THWrapper::Tensor<TensorFloat>::size(const THFloatTensor *tensor)
{
    return THFloatTensor_newSizeOf((THFloatTensor*)tensor);
}

template<>
long THWrapper::Tensor<TensorFloat>::size(const THFloatTensor *tensor, int dim)
{
    return THFloatTensor_size(tensor, dim);
}

template<>
THLongStorage* THWrapper::Tensor<TensorFloat>::stride(const THFloatTensor *tensor)
{
    return THFloatTensor_newStrideOf((THFloatTensor*)tensor);
}

template<>
float *THWrapper::Tensor<TensorFloat>::data(const THFloatTensor *tensor)
{
    return THFloatTensor_data((THFloatTensor*)tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
int THWrapper::Tensor<TensorFloat>::isContiguous(const THFloatTensor *tensor)
{
    return THFloatTensor_isContiguous(tensor);
}

template<>
int THWrapper::Tensor<TensorFloat>::nElement(const THFloatTensor *tensor)
{
    return THFloatTensor_nElement(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
void THWrapper::Tensor<TensorFloat>::narrow(THFloatTensor *self, THFloatTensor *src, int dimension, long firstIndex, long size)
{
    THFloatTensor_narrow(self, src, dimension, firstIndex, size);
}

template<>
void THWrapper::Tensor<TensorFloat>::select(THFloatTensor *self, THFloatTensor *src, int dimension, long sliceIndex)
{
    THFloatTensor_select(self, src, dimension, sliceIndex);
}

template<>
void THWrapper::Tensor<TensorFloat>::transpose(THFloatTensor *self, THFloatTensor *src, int dimension1, int dimension2)
{
    THFloatTensor_transpose(self, src, dimension1, dimension2);
}

//////////////////////////////////////////////////////////////////////////

template<>
void THWrapper::Tensor<TensorFloat>::fill(THFloatTensor *r, float val)
{
    return THFloatTensor_fill(r, val);
}

template<>
float THWrapper::Tensor<TensorFloat>::minall(THFloatTensor *r)
{
    return THFloatTensor_minall(r);
}

template<>
float THWrapper::Tensor<TensorFloat>::maxall(THFloatTensor *r)
{
    return THFloatTensor_maxall(r);
}

template<>
void THWrapper::Tensor<TensorFloat>::max(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    THLongTensor *l = THLongTensor_new();
    THFloatTensor_max(values, l, t, dimension);
    THLongTensor_free(l);
}

template<>
void THWrapper::Tensor<TensorFloat>::sum(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    return THFloatTensor_sum(values, t, dimension);
}

template<>
void THWrapper::Tensor<TensorFloat>::add(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_add(r, t, val);
}

template<>
void THWrapper::Tensor<TensorFloat>::cadd(THFloatTensor *r, THFloatTensor *t, float val, THFloatTensor *src)
{
    THFloatTensor_cadd(r, t, val, src);
}

template<>
void THWrapper::Tensor<TensorFloat>::mul(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_mul(r, t, val);
}

template<>
void THWrapper::Tensor<TensorFloat>::cmul(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cmul(r, t, src);
}

template<>
void THWrapper::Tensor<TensorFloat>::cdiv(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cdiv(r, t, src);
}

template<>
void THWrapper::Tensor<TensorFloat>::pow(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_pow(r, t, val);
}

template<>
void THWrapper::Tensor<TensorFloat>::cpow(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cpow(r, t, src);
}

template<>
void THWrapper::Tensor<TensorFloat>::addmv(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat, THFloatTensor *vec)
{
    THFloatTensor_addmv(r, beta, t, alpha, mat, vec);
}

template<>
void THWrapper::Tensor<TensorFloat>::addmm(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)
{
    THFloatTensor_addmm(r, beta, t, alpha, mat1, mat2);
}

template<>
void THWrapper::Tensor<TensorFloat>::addr(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *vec1, THFloatTensor *vec2)
{
    THFloatTensor_addr(r, beta, t, alpha, vec1, vec2);
}

template<>
void THWrapper::Tensor<TensorFloat>::abs(THFloatTensor *r, THFloatTensor *t)
{
    THFloatTensor_abs(r, t);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<>
void THWrapper::NN<TensorFloat>::BatchNormalization_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *running_mean, THFloatTensor *running_var,
    THFloatTensor *save_mean, THFloatTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_FloatBatchNormalization_updateOutput(state, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}

template<>
void THWrapper::NN<TensorFloat>::SpatialAveragePooling_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_FloatSpatialAveragePooling_updateOutput(state, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}

template<>
void THWrapper::NN<TensorFloat>::SpatialConvolutionMM_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *finput, THFloatTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_FloatSpatialConvolutionMM_updateOutput(state, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}

template<>
void THWrapper::NN<TensorFloat>::SpatialMaxPooling_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output, THFloatTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_FloatSpatialMaxPooling_updateOutput(state, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}

template<>
void THWrapper::NN<TensorFloat>::SpatialReflectionPadding_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_FloatSpatialReflectionPadding_updateOutput(state, input, output, pad_l, pad_r, pad_t, pad_b);
}

template<>
void THWrapper::NN<TensorFloat>::Square_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output)
{
    THNN_FloatSquare_updateOutput(state, input, output);
}

template<>
void THWrapper::NN<TensorFloat>::Sqrt_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output, float eps)
{
    THNN_FloatSqrt_updateOutput(state, input, output, eps);
}

template<>
void THWrapper::NN<TensorFloat>::Threshold_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    float threshold, float val, bool inplace)
{
    THNN_FloatThreshold_updateOutput(state, input, output, threshold, val, inplace);
}
