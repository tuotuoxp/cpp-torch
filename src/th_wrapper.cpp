#include "../include/th_wrapper.h"
#include "../include/torch/Storage.h"
#include "../include/torch/Tensor.h"

#include <TH/TH.h>
#include <THNN/THNN.h>
#include <assert.h>


namespace cpptorch { namespace th {

template<>
THLongStorage* Storage<long>::newWithAllocator(THAllocator *allocator, void *allocatorContext)
{
    return THLongStorage_newWithAllocator(0, allocator, allocatorContext);
}
template<>
THFloatStorage* Storage<float>::newWithAllocator(THAllocator *allocator, void *allocatorContext)
{
    return THFloatStorage_newWithAllocator(0, allocator, allocatorContext);
}
template<>
THDoubleStorage* Storage<double>::newWithAllocator(THAllocator *allocator, void *allocatorContext)
{
    return THDoubleStorage_newWithAllocator(0, allocator, allocatorContext);
}

template<>
THLongStorage* Storage<long>::newWithDataAndAllocator(long *data, long size, THAllocator *allocator, void *allocatorContext)
{
    return THLongStorage_newWithDataAndAllocator(data, size, allocator, allocatorContext);
}
template<>
THFloatStorage* Storage<float>::newWithDataAndAllocator(float *data, long size, THAllocator *allocator, void *allocatorContext)
{
    return THFloatStorage_newWithDataAndAllocator(data, size, allocator, allocatorContext);
}
template<>
THDoubleStorage* Storage<double>::newWithDataAndAllocator(double *data, long size, THAllocator *allocator, void *allocatorContext)
{
    return THDoubleStorage_newWithDataAndAllocator(data, size, allocator, allocatorContext);
}

template<>
void Storage<long>::retain(THLongStorage *storage)
{
    THLongStorage_retain(storage);
}
template<>
void Storage<float>::retain(THFloatStorage *storage)
{
    THFloatStorage_retain(storage);
}
template<>
void Storage<double>::retain(THDoubleStorage *storage)
{
    THDoubleStorage_retain(storage);
}

template<>
void Storage<long>::release(THLongStorage *storage)
{
    THLongStorage_free(storage);
}
template<>
void Storage<float>::release(THFloatStorage *storage)
{
    THFloatStorage_free(storage);
}
template<>
void Storage<double>::release(THDoubleStorage *storage)
{
    THDoubleStorage_free(storage);
}

//////////////////////////////////////////////////////////////////////////

template<>
long* Storage<long>::data(THLongStorage *storage)
{
    return THLongStorage_data(storage);
}
template<>
float* Storage<float>::data(THFloatStorage *storage)
{
    return THFloatStorage_data(storage);
}
template<>
double* Storage<double>::data(THDoubleStorage *storage)
{
    return THDoubleStorage_data(storage);
}

template<>
int Storage<long>::size(THLongStorage *storage)
{
    return THLongStorage_size(storage);
}
template<>
int Storage<float>::size(THFloatStorage *storage)
{
    return THFloatStorage_size(storage);
}
template<>
int Storage<double>::size(THDoubleStorage *storage)
{
    return THDoubleStorage_size(storage);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
THLongTensor* Tensor<long>::newWithStorage(THLongStorage *storage, long offset, THLongStorage *size, THLongStorage *stride)
{
    return THLongTensor_newWithStorage(storage, offset, size, stride);
}
template<>
THFloatTensor* Tensor<float>::newWithStorage(THFloatStorage *storage, long offset, THLongStorage *size, THLongStorage *stride)
{
    return THFloatTensor_newWithStorage(storage, offset, size, stride);
}
template<>
THDoubleTensor* Tensor<double>::newWithStorage(THDoubleStorage *storage, long offset, THLongStorage *size, THLongStorage *stride)
{
    return THDoubleTensor_newWithStorage(storage, offset, size, stride);
}

template<>
void Tensor<long>::resize(THLongTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THLongTensor_resize(self, size, stride);
}
template<>
void Tensor<float>::resize(THFloatTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THFloatTensor_resize(self, size, stride);
}
template<>
void Tensor<double>::resize(THDoubleTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THDoubleTensor_resize(self, size, stride);
}

template<>
void Tensor<long>::resizeAs(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_resizeAs(self, src);
}
template<>
void Tensor<float>::resizeAs(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_resizeAs(self, src);
}
template<>
void Tensor<double>::resizeAs(THDoubleTensor *self, THDoubleTensor *src)
{
    THDoubleTensor_resizeAs(self, src);
}

template<>
void Tensor<long>::copy(THLongTensor *self, THLongTensor *src)
{
    THLongTensor_copy(self, src);
}
template<>
void Tensor<float>::copy(THFloatTensor *self, THFloatTensor *src)
{
    THFloatTensor_copy(self, src);
}
template<>
void Tensor<double>::copy(THDoubleTensor *self, THDoubleTensor *src)
{
    THDoubleTensor_copy(self, src);
}

template<>
void Tensor<long>::retain(THLongTensor *tensor)
{
    THLongTensor_retain(tensor);
}
template<>
void Tensor<float>::retain(THFloatTensor *tensor)
{
    THFloatTensor_retain(tensor);
}
template<>
void Tensor<double>::retain(THDoubleTensor *tensor)
{
    THDoubleTensor_retain(tensor);
}

template<>
void Tensor<long>::release(THLongTensor *tensor)
{
    THLongTensor_free(tensor);
}
template<>
void Tensor<float>::release(THFloatTensor *tensor)
{
    THFloatTensor_free(tensor);
}
template<>
void Tensor<double>::release(THDoubleTensor *tensor)
{
    THDoubleTensor_free(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
THLongStorage* Tensor<long>::storage(const THLongTensor *tensor)
{
    return THLongTensor_storage(tensor);
}
template<>
THFloatStorage* Tensor<float>::storage(const THFloatTensor *tensor)
{
    return THFloatTensor_storage(tensor);
}
template<>
THDoubleStorage* Tensor<double>::storage(const THDoubleTensor *tensor)
{
    return THDoubleTensor_storage(tensor);
}

template<>
long Tensor<long>::storageOffset(const THLongTensor *tensor)
{
    return THLongTensor_storageOffset(tensor);
}
template<>
long Tensor<float>::storageOffset(const THFloatTensor *tensor)
{
    return THFloatTensor_storageOffset(tensor);
}
template<>
long Tensor<double>::storageOffset(const THDoubleTensor *tensor)
{
    return THDoubleTensor_storageOffset(tensor);
}

template<>
int Tensor<long>::nDimension(const THLongTensor *tensor)
{
    return THLongTensor_nDimension(tensor);
}
template<>
int Tensor<float>::nDimension(const THFloatTensor *tensor)
{
    return THFloatTensor_nDimension(tensor);
}
template<>
int Tensor<double>::nDimension(const THDoubleTensor *tensor)
{
    return THDoubleTensor_nDimension(tensor);
}

template<>
THLongStorage* Tensor<long>::size(const THLongTensor *tensor)
{
    return THLongTensor_newSizeOf((THLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float>::size(const THFloatTensor *tensor)
{
    return THFloatTensor_newSizeOf((THFloatTensor*)tensor);
}
template<>
THLongStorage* Tensor<double>::size(const THDoubleTensor *tensor)
{
    return THDoubleTensor_newSizeOf((THDoubleTensor*)tensor);
}

template<>
long Tensor<long>::size(const THLongTensor *tensor, int dim)
{
    return THLongTensor_size(tensor, dim);
}
template<>
long Tensor<float>::size(const THFloatTensor *tensor, int dim)
{
    return THFloatTensor_size(tensor, dim);
}
template<>
long Tensor<double>::size(const THDoubleTensor *tensor, int dim)
{
    return THDoubleTensor_size(tensor, dim);
}

template<>
THLongStorage* Tensor<long>::stride(const THLongTensor *tensor)
{
    return THLongTensor_newStrideOf((THLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float>::stride(const THFloatTensor *tensor)
{
    return THFloatTensor_newStrideOf((THFloatTensor*)tensor);
}
template<>
THLongStorage* Tensor<double>::stride(const THDoubleTensor *tensor)
{
    return THDoubleTensor_newStrideOf((THDoubleTensor*)tensor);
}

template<>
long *Tensor<long>::data(const THLongTensor *tensor)
{
    return THLongTensor_data((THLongTensor*)tensor);
}
template<>
float *Tensor<float>::data(const THFloatTensor *tensor)
{
    return THFloatTensor_data((THFloatTensor*)tensor);
}
template<>
double *Tensor<double>::data(const THDoubleTensor *tensor)
{
    return THDoubleTensor_data((THDoubleTensor*)tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
int Tensor<long>::isContiguous(const THLongTensor *tensor)
{
    return THLongTensor_isContiguous(tensor);
}
template<>
int Tensor<float>::isContiguous(const THFloatTensor *tensor)
{
    return THFloatTensor_isContiguous(tensor);
}
template<>
int Tensor<double>::isContiguous(const THDoubleTensor *tensor)
{
    return THDoubleTensor_isContiguous(tensor);
}

template<>
int Tensor<long>::nElement(const THLongTensor *tensor)
{
    return THLongTensor_nElement(tensor);
}
template<>
int Tensor<float>::nElement(const THFloatTensor *tensor)
{
    return THFloatTensor_nElement(tensor);
}
template<>
int Tensor<double>::nElement(const THDoubleTensor *tensor)
{
    return THDoubleTensor_nElement(tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long>::narrow(THLongTensor *self, THLongTensor *src, int dimension, long firstIndex, long size)
{
    THLongTensor_narrow(self, src, dimension, firstIndex, size);
}
template<>
void Tensor<float>::narrow(THFloatTensor *self, THFloatTensor *src, int dimension, long firstIndex, long size)
{
    THFloatTensor_narrow(self, src, dimension, firstIndex, size);
}
template<>
void Tensor<double>::narrow(THDoubleTensor *self, THDoubleTensor *src, int dimension, long firstIndex, long size)
{
    THDoubleTensor_narrow(self, src, dimension, firstIndex, size);
}

template<>
void Tensor<long>::select(THLongTensor *self, THLongTensor *src, int dimension, long sliceIndex)
{
    THLongTensor_select(self, src, dimension, sliceIndex);
}
template<>
void Tensor<float>::select(THFloatTensor *self, THFloatTensor *src, int dimension, long sliceIndex)
{
    THFloatTensor_select(self, src, dimension, sliceIndex);
}
template<>
void Tensor<double>::select(THDoubleTensor *self, THDoubleTensor *src, int dimension, long sliceIndex)
{
    THDoubleTensor_select(self, src, dimension, sliceIndex);
}

template<>
void Tensor<long>::transpose(THLongTensor *self, THLongTensor *src, int dimension1, int dimension2)
{
    THLongTensor_transpose(self, src, dimension1, dimension2);
}
template<>
void Tensor<float>::transpose(THFloatTensor *self, THFloatTensor *src, int dimension1, int dimension2)
{
    THFloatTensor_transpose(self, src, dimension1, dimension2);
}
template<>
void Tensor<double>::transpose(THDoubleTensor *self, THDoubleTensor *src, int dimension1, int dimension2)
{
    THDoubleTensor_transpose(self, src, dimension1, dimension2);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long>::fill(THLongTensor *r, long val)
{
    return THLongTensor_fill(r, val);
}
template<>
void Tensor<float>::fill(THFloatTensor *r, float val)
{
    return THFloatTensor_fill(r, val);
}
template<>
void Tensor<double>::fill(THDoubleTensor *r, double val)
{
    return THDoubleTensor_fill(r, val);
}

template<>
long Tensor<long>::minall(THLongTensor *r)
{
    return THLongTensor_minall(r);
}
template<>
float Tensor<float>::minall(THFloatTensor *r)
{
    return THFloatTensor_minall(r);
}
template<>
double Tensor<double>::minall(THDoubleTensor *r)
{
    return THDoubleTensor_minall(r);
}

template<>
long Tensor<long>::maxall(THLongTensor *r)
{
    return THLongTensor_maxall(r);
}
template<>
float Tensor<float>::maxall(THFloatTensor *r)
{
    return THFloatTensor_maxall(r);
}
template<>
double Tensor<double>::maxall(THDoubleTensor *r)
{
    return THDoubleTensor_maxall(r);
}

template<>
void Tensor<long>::max(THLongTensor *values, THLongTensor *t, int dimension)
{
    THLongTensor *l = THLongTensor_new();
    THLongTensor_max(values, l, t, dimension);
    THLongTensor_free(l);
}
template<>
void Tensor<float>::max(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    THLongTensor *l = THLongTensor_new();
    THFloatTensor_max(values, l, t, dimension);
    THLongTensor_free(l);
}
template<>
void Tensor<double>::max(THDoubleTensor *values, THDoubleTensor *t, int dimension)
{
    THLongTensor *l = THLongTensor_new();
    THDoubleTensor_max(values, l, t, dimension);
    THLongTensor_free(l);
}

template<>
void Tensor<long>::sum(THLongTensor *values, THLongTensor *t, int dimension)
{
    return THLongTensor_sum(values, t, dimension);
}
template<>
void Tensor<float>::sum(THFloatTensor *values, THFloatTensor *t, int dimension)
{
    return THFloatTensor_sum(values, t, dimension);
}
template<>
void Tensor<double>::sum(THDoubleTensor *values, THDoubleTensor *t, int dimension)
{
    return THDoubleTensor_sum(values, t, dimension);
}

template<>
void Tensor<long>::add(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_add(r, t, val);
}
template<>
void Tensor<float>::add(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_add(r, t, val);
}
template<>
void Tensor<double>::add(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_add(r, t, val);
}

template<>
void Tensor<long>::cadd(THLongTensor *r, THLongTensor *t, long val, THLongTensor *src)
{
    THLongTensor_cadd(r, t, val, src);
}
template<>
void Tensor<float>::cadd(THFloatTensor *r, THFloatTensor *t, float val, THFloatTensor *src)
{
    THFloatTensor_cadd(r, t, val, src);
}
template<>
void Tensor<double>::cadd(THDoubleTensor *r, THDoubleTensor *t, double val, THDoubleTensor *src)
{
    THDoubleTensor_cadd(r, t, val, src);
}

template<>
void Tensor<long>::mul(THLongTensor *r, THLongTensor *t, long val)
{
    THLongTensor_mul(r, t, val);
}
template<>
void Tensor<float>::mul(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_mul(r, t, val);
}
template<>
void Tensor<double>::mul(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_mul(r, t, val);
}

template<>
void Tensor<long>::cmul(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cmul(r, t, src);
}
template<>
void Tensor<float>::cmul(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cmul(r, t, src);
}
template<>
void Tensor<double>::cmul(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cmul(r, t, src);
}

template<>
void Tensor<long>::cdiv(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cdiv(r, t, src);
}
template<>
void Tensor<float>::cdiv(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cdiv(r, t, src);
}
template<>
void Tensor<double>::cdiv(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cdiv(r, t, src);
}

template<>
void Tensor<long>::pow(THLongTensor *r, THLongTensor *t, long val)
{
    //THLongTensor_pow(r, t, val);
    assert(0);
}
template<>
void Tensor<float>::pow(THFloatTensor *r, THFloatTensor *t, float val)
{
    THFloatTensor_pow(r, t, val);
}
template<>
void Tensor<double>::pow(THDoubleTensor *r, THDoubleTensor *t, double val)
{
    THDoubleTensor_pow(r, t, val);
}

template<>
void Tensor<long>::cpow(THLongTensor *r, THLongTensor *t, THLongTensor *src)
{
    THLongTensor_cpow(r, t, src);
}
template<>
void Tensor<float>::cpow(THFloatTensor *r, THFloatTensor *t, THFloatTensor *src)
{
    THFloatTensor_cpow(r, t, src);
}
template<>
void Tensor<double>::cpow(THDoubleTensor *r, THDoubleTensor *t, THDoubleTensor *src)
{
    THDoubleTensor_cpow(r, t, src);
}

template<>
void Tensor<long>::addmv(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat, THLongTensor *vec)
{
    THLongTensor_addmv(r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<float>::addmv(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat, THFloatTensor *vec)
{
    THFloatTensor_addmv(r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<double>::addmv(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat, THDoubleTensor *vec)
{
    THDoubleTensor_addmv(r, beta, t, alpha, mat, vec);
}

template<>
void Tensor<long>::addmm(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *mat1, THLongTensor *mat2)
{
    THLongTensor_addmm(r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<float>::addmm(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)
{
    THFloatTensor_addmm(r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<double>::addmm(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat1, THDoubleTensor *mat2)
{
    THDoubleTensor_addmm(r, beta, t, alpha, mat1, mat2);
}

template<>
void Tensor<long>::addr(THLongTensor *r, long beta, THLongTensor *t, long alpha, THLongTensor *vec1, THLongTensor *vec2)
{
    THLongTensor_addr(r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<float>::addr(THFloatTensor *r, float beta, THFloatTensor *t, float alpha, THFloatTensor *vec1, THFloatTensor *vec2)
{
    THFloatTensor_addr(r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<double>::addr(THDoubleTensor *r, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *vec1, THDoubleTensor *vec2)
{
    THDoubleTensor_addr(r, beta, t, alpha, vec1, vec2);
}

template<>
void Tensor<long>::abs(THLongTensor *r, THLongTensor *t)
{
    THLongTensor_abs(r, t);
}
template<>
void Tensor<float>::abs(THFloatTensor *r, THFloatTensor *t)
{
    THFloatTensor_abs(r, t);
}
template<>
void Tensor<double>::abs(THDoubleTensor *r, THDoubleTensor *t)
{
    THDoubleTensor_abs(r, t);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
void NN<float>::BatchNormalization_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *running_mean, THFloatTensor *running_var,
    THFloatTensor *save_mean, THFloatTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_FloatBatchNormalization_updateOutput(state, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}
template<>
void NN<double>::BatchNormalization_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    THDoubleTensor *weight, THDoubleTensor *bias, THDoubleTensor *running_mean, THDoubleTensor *running_var,
    THDoubleTensor *save_mean, THDoubleTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_DoubleBatchNormalization_updateOutput(state, input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}

template<>
void NN<float>::SpatialAveragePooling_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_FloatSpatialAveragePooling_updateOutput(state, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}
template<>
void NN<double>::SpatialAveragePooling_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_DoubleSpatialAveragePooling_updateOutput(state, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}

template<>
void NN<float>::SpatialConvolutionMM_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *finput, THFloatTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_FloatSpatialConvolutionMM_updateOutput(state, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}
template<>
void NN<double>::SpatialConvolutionMM_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    THDoubleTensor *weight, THDoubleTensor *bias, THDoubleTensor *finput, THDoubleTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_DoubleSpatialConvolutionMM_updateOutput(state, input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}

template<>
void NN<float>::SpatialMaxPooling_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output, THFloatTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_FloatSpatialMaxPooling_updateOutput(state, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}
template<>
void NN<double>::SpatialMaxPooling_updateOutput(THNNState *state,
    THDoubleTensor *input, THDoubleTensor *output, THDoubleTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_DoubleSpatialMaxPooling_updateOutput(state, input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}

template<>
void NN<float>::SpatialReflectionPadding_updateOutput(THNNState *state,
    THFloatTensor *input, THFloatTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_FloatSpatialReflectionPadding_updateOutput(state, input, output, pad_l, pad_r, pad_t, pad_b);
}
template<>
void NN<double>::SpatialReflectionPadding_updateOutput(THNNState *state,
    THDoubleTensor *input, THDoubleTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_DoubleSpatialReflectionPadding_updateOutput(state, input, output, pad_l, pad_r, pad_t, pad_b);
}

template<>
void NN<float>::Square_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output)
{
    THNN_FloatSquare_updateOutput(state, input, output);
}
template<>
void NN<double>::Square_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output)
{
    THNN_DoubleSquare_updateOutput(state, input, output);
}

template<>
void NN<float>::Sqrt_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output, float eps)
{
    THNN_FloatSqrt_updateOutput(state, input, output, eps);
}
template<>
void NN<double>::Sqrt_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output, double eps)
{
    THNN_DoubleSqrt_updateOutput(state, input, output, eps);
}

template<>
void NN<float>::Threshold_updateOutput(THNNState *state, THFloatTensor *input, THFloatTensor *output,
    float threshold, float val, bool inplace)
{
    THNN_FloatThreshold_updateOutput(state, input, output, threshold, val, inplace);
}
template<>
void NN<double>::Threshold_updateOutput(THNNState *state, THDoubleTensor *input, THDoubleTensor *output,
    double threshold, double val, bool inplace)
{
    THNN_DoubleThreshold_updateOutput(state, input, output, threshold, val, inplace);
}


}
}
