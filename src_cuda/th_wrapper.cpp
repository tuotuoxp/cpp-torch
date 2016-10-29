#include "../src/th_wrapper.h"
#include "../include/torch/Storage.h"
#include "../include/torch/Tensor.h"

#include <TH/TH.h>
#include <THC/THC.h>
#include <assert.h>


extern THCState* GetCudaState();


namespace cpptorch { namespace th {


template <>
THCudaLongStorage* Storage<long, GPU_Cuda>::newWithData(const long *ptr_src, long count, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        cpptorch::Storage<long> th;
        th.unserialze(ptr_src, count, take_ownership_of_data);
        THCudaLongStorage *thc = THCudaLongStorage_newWithSize(GetCudaState(), count);
        THCudaLongStorage_copyCPU(GetCudaState(), thc, th);
        return thc;
    }
    else
    {
        return THCudaLongStorage_new(GetCudaState());
    }
}
template <>
THCudaStorage* Storage<float, GPU_Cuda>::newWithData(const float *ptr_src, long count, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        cpptorch::Storage<float> th;
        th.unserialze(ptr_src, count, take_ownership_of_data);
        THCudaStorage *thc = THCudaStorage_newWithSize(GetCudaState(), count);
        THCudaStorage_copyCPU(GetCudaState(), thc, th);
        return thc;
    }
    else
    {
        return THCudaStorage_new(GetCudaState());
    }
}
template <>
THCudaDoubleStorage* Storage<double, GPU_Cuda>::newWithData(const double *ptr_src, long count, bool take_ownership_of_data)
{
    if (ptr_src)
    {
        cpptorch::Storage<double> th;
        th.unserialze(ptr_src, count, take_ownership_of_data);
        THCudaDoubleStorage *thc = THCudaDoubleStorage_newWithSize(GetCudaState(), count);
        THCudaDoubleStorage_copyCPU(GetCudaState(), thc, th);
        return thc;
    }
    else
    {
        return THCudaDoubleStorage_new(GetCudaState());
    }
}

template<>
void Storage<long, GPU_Cuda>::retain(THCudaLongStorage *storage)
{
    THCudaLongStorage_retain(GetCudaState(), storage);
}
template<>
void Storage<float, GPU_Cuda>::retain(THCudaStorage *storage)
{
    THCudaStorage_retain(GetCudaState(), storage);
}
template<>
void Storage<double, GPU_Cuda>::retain(THCudaDoubleStorage *storage)
{
    THCudaDoubleStorage_retain(GetCudaState(), storage);
}

template<>
void Storage<long, GPU_Cuda>::release(THCudaLongStorage *storage)
{
    THCudaLongStorage_free(GetCudaState(), storage);
}
template<>
void Storage<float, GPU_Cuda>::release(THCudaStorage *storage)
{
    THCudaStorage_free(GetCudaState(), storage);
}
template<>
void Storage<double, GPU_Cuda>::release(THCudaDoubleStorage *storage)
{
    THCudaDoubleStorage_free(GetCudaState(), storage);
}

//////////////////////////////////////////////////////////////////////////

template<>
long* Storage<long, GPU_Cuda>::data(THCudaLongStorage *storage)
{
    return THCudaLongStorage_data(GetCudaState(), storage);
}
template<>
float* Storage<float, GPU_Cuda>::data(THCudaStorage *storage)
{
    return THCudaStorage_data(GetCudaState(), storage);
}
template<>
double* Storage<double, GPU_Cuda>::data(THCudaDoubleStorage *storage)
{
    return THCudaDoubleStorage_data(GetCudaState(), storage);
}

template<>
long Storage<long, GPU_Cuda>::size(THCudaLongStorage *storage)
{
    return (long)THCudaLongStorage_size(GetCudaState(), storage);
}
template<>
long Storage<float, GPU_Cuda>::size(THCudaStorage *storage)
{
    return (long)THCudaStorage_size(GetCudaState(), storage);
}
template<>
long Storage<double, GPU_Cuda>::size(THCudaDoubleStorage *storage)
{
    return (long)THCudaDoubleStorage_size(GetCudaState(), storage);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<>
THCudaLongTensor* Tensor<long, GPU_Cuda>::newWithStorage(THCudaLongStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THCudaLongTensor_newWithStorage(GetCudaState(), storage, offset, nullptr, nullptr);
    case 1: return THCudaLongTensor_newWithStorage1d(GetCudaState(), storage, offset, size[0], stride[0]);
    case 2: return THCudaLongTensor_newWithStorage2d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THCudaLongTensor_newWithStorage3d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THCudaLongTensor_newWithStorage4d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THCudaLongTensor_newWithStorage(GetCudaState(), storage, offset, cpptorch::Storage<long>(size, dim, false), cpptorch::Storage<long>(stride, dim, false));
}
template<>
THCudaTensor* Tensor<float, GPU_Cuda>::newWithStorage(THCudaStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THCudaTensor_newWithStorage(GetCudaState(), storage, offset, nullptr, nullptr);
    case 1: return THCudaTensor_newWithStorage1d(GetCudaState(), storage, offset, size[0], stride[0]);
    case 2: return THCudaTensor_newWithStorage2d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THCudaTensor_newWithStorage3d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THCudaTensor_newWithStorage4d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THCudaTensor_newWithStorage(GetCudaState(), storage, offset, cpptorch::Storage<long>(size, dim, false), cpptorch::Storage<long>(stride, dim, false));
}
template<>
THCudaDoubleTensor* Tensor<double, GPU_Cuda>::newWithStorage(THCudaDoubleStorage *storage, long offset, int dim, const long *size, const long *stride)
{
    switch (dim)
    {
    case 0: return THCudaDoubleTensor_newWithStorage(GetCudaState(), storage, offset, nullptr, nullptr);
    case 1: return THCudaDoubleTensor_newWithStorage1d(GetCudaState(), storage, offset, size[0], stride[0]);
    case 2: return THCudaDoubleTensor_newWithStorage2d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1]);
    case 3: return THCudaDoubleTensor_newWithStorage3d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
    case 4: return THCudaDoubleTensor_newWithStorage4d(GetCudaState(), storage, offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
    }
    return THCudaDoubleTensor_newWithStorage(GetCudaState(), storage, offset, cpptorch::Storage<long>(size, dim, false), cpptorch::Storage<long>(stride, dim, false));
}

template<>
void Tensor<long, GPU_Cuda>::resize(THCudaLongTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THCudaLongTensor_resize(GetCudaState(), self, size, stride);
}
template<>
void Tensor<float, GPU_Cuda>::resize(THCudaTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THCudaTensor_resize(GetCudaState(), self, size, stride);
}
template<>
void Tensor<double, GPU_Cuda>::resize(THCudaDoubleTensor *self, THLongStorage *size, THLongStorage *stride)
{
    THCudaDoubleTensor_resize(GetCudaState(), self, size, stride);
}

template<>
void Tensor<long, GPU_Cuda>::resizeAs(THCudaLongTensor *self, THCudaLongTensor *src)
{
    THCudaLongTensor_resizeAs(GetCudaState(), self, src);
}
template<>
void Tensor<float, GPU_Cuda>::resizeAs(THCudaTensor *self, THCudaTensor *src)
{
    THCudaTensor_resizeAs(GetCudaState(), self, src);
}
template<>
void Tensor<double, GPU_Cuda>::resizeAs(THCudaDoubleTensor *self, THCudaDoubleTensor *src)
{
    THCudaDoubleTensor_resizeAs(GetCudaState(), self, src);
}

template<>
void Tensor<long, GPU_Cuda>::copy(THCudaLongTensor *self, THCudaLongTensor *src)
{
    THCudaLongTensor_copy(GetCudaState(), self, src);
}
template<>
void Tensor<float, GPU_Cuda>::copy(THCudaTensor *self, THCudaTensor *src)
{
    THCudaTensor_copy(GetCudaState(), self, src);
}
template<>
void Tensor<double, GPU_Cuda>::copy(THCudaDoubleTensor *self, THCudaDoubleTensor *src)
{
    THCudaDoubleTensor_copy(GetCudaState(), self, src);
}

template<>
void Tensor<long, GPU_Cuda>::retain(THCudaLongTensor *tensor)
{
    THCudaLongTensor_retain(GetCudaState(), tensor);
}
template<>
void Tensor<float, GPU_Cuda>::retain(THCudaTensor *tensor)
{
    THCudaTensor_retain(GetCudaState(), tensor);
}
template<>
void Tensor<double, GPU_Cuda>::retain(THCudaDoubleTensor *tensor)
{
    THCudaDoubleTensor_retain(GetCudaState(), tensor);
}

template<>
void Tensor<long, GPU_Cuda>::release(THCudaLongTensor *tensor)
{
    THCudaLongTensor_free(GetCudaState(), tensor);
}
template<>
void Tensor<float, GPU_Cuda>::release(THCudaTensor *tensor)
{
    THCudaTensor_free(GetCudaState(), tensor);
}
template<>
void Tensor<double, GPU_Cuda>::release(THCudaDoubleTensor *tensor)
{
    THCudaDoubleTensor_free(GetCudaState(), tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
THCudaLongStorage* Tensor<long, GPU_Cuda>::storage(const THCudaLongTensor *tensor)
{
    return THCudaLongTensor_storage(GetCudaState(), tensor);
}
template<>
THCudaStorage* Tensor<float, GPU_Cuda>::storage(const THCudaTensor *tensor)
{
    return THCudaTensor_storage(GetCudaState(), tensor);
}
template<>
THCudaDoubleStorage* Tensor<double, GPU_Cuda>::storage(const THCudaDoubleTensor *tensor)
{
    return THCudaDoubleTensor_storage(GetCudaState(), tensor);
}

template<>
long Tensor<long, GPU_Cuda>::storageOffset(const THCudaLongTensor *tensor)
{
    return (long)THCudaLongTensor_storageOffset(GetCudaState(), tensor);
}
template<>
long Tensor<float, GPU_Cuda>::storageOffset(const THCudaTensor *tensor)
{
    return (long)THCudaTensor_storageOffset(GetCudaState(), tensor);
}
template<>
long Tensor<double, GPU_Cuda>::storageOffset(const THCudaDoubleTensor *tensor)
{
    return (long)THCudaDoubleTensor_storageOffset(GetCudaState(), tensor);
}

template<>
int Tensor<long, GPU_Cuda>::nDimension(const THCudaLongTensor *tensor)
{
    return THCudaLongTensor_nDimension(GetCudaState(), tensor);
}
template<>
int Tensor<float, GPU_Cuda>::nDimension(const THCudaTensor *tensor)
{
    return THCudaTensor_nDimension(GetCudaState(), tensor);
}
template<>
int Tensor<double, GPU_Cuda>::nDimension(const THCudaDoubleTensor *tensor)
{
    return THCudaDoubleTensor_nDimension(GetCudaState(), tensor);
}

template<>
THLongStorage* Tensor<long, GPU_Cuda>::size(const THCudaLongTensor *tensor)
{
    return THCudaLongTensor_newSizeOf(GetCudaState(), (THCudaLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float, GPU_Cuda>::size(const THCudaTensor *tensor)
{
    return THCudaTensor_newSizeOf(GetCudaState(), (THCudaTensor*)tensor);
}
template<>
THLongStorage* Tensor<double, GPU_Cuda>::size(const THCudaDoubleTensor *tensor)
{
    return THCudaDoubleTensor_newSizeOf(GetCudaState(), (THCudaDoubleTensor*)tensor);
}

template<>
long Tensor<long, GPU_Cuda>::size(const THCudaLongTensor *tensor, int dim)
{
    return THCudaLongTensor_size(GetCudaState(), tensor, dim);
}
template<>
long Tensor<float, GPU_Cuda>::size(const THCudaTensor *tensor, int dim)
{
    return THCudaTensor_size(GetCudaState(), tensor, dim);
}
template<>
long Tensor<double, GPU_Cuda>::size(const THCudaDoubleTensor *tensor, int dim)
{
    return THCudaDoubleTensor_size(GetCudaState(), tensor, dim);
}

template<>
THLongStorage* Tensor<long, GPU_Cuda>::stride(const THCudaLongTensor *tensor)
{
    return THCudaLongTensor_newStrideOf(GetCudaState(), (THCudaLongTensor*)tensor);
}
template<>
THLongStorage* Tensor<float, GPU_Cuda>::stride(const THCudaTensor *tensor)
{
    return THCudaTensor_newStrideOf(GetCudaState(), (THCudaTensor*)tensor);
}
template<>
THLongStorage* Tensor<double, GPU_Cuda>::stride(const THCudaDoubleTensor *tensor)
{
    return THCudaDoubleTensor_newStrideOf(GetCudaState(), (THCudaDoubleTensor*)tensor);
}

template<>
long *Tensor<long, GPU_Cuda>::data(const THCudaLongTensor *tensor)
{
    return THCudaLongTensor_data(GetCudaState(), (THCudaLongTensor*)tensor);
}
template<>
float *Tensor<float, GPU_Cuda>::data(const THCudaTensor *tensor)
{
    return THCudaTensor_data(GetCudaState(), (THCudaTensor*)tensor);
}
template<>
double *Tensor<double, GPU_Cuda>::data(const THCudaDoubleTensor *tensor)
{
    return THCudaDoubleTensor_data(GetCudaState(), (THCudaDoubleTensor*)tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
int Tensor<long, GPU_Cuda>::isContiguous(const THCudaLongTensor *tensor)
{
    return THCudaLongTensor_isContiguous(GetCudaState(), tensor);
}
template<>
int Tensor<float, GPU_Cuda>::isContiguous(const THCudaTensor *tensor)
{
    return THCudaTensor_isContiguous(GetCudaState(), tensor);
}
template<>
int Tensor<double, GPU_Cuda>::isContiguous(const THCudaDoubleTensor *tensor)
{
    return THCudaDoubleTensor_isContiguous(GetCudaState(), tensor);
}

template<>
long Tensor<long, GPU_Cuda>::nElement(const THCudaLongTensor *tensor)
{
    return (long)THCudaLongTensor_nElement(GetCudaState(), tensor);
}
template<>
long Tensor<float, GPU_Cuda>::nElement(const THCudaTensor *tensor)
{
    return (long)THCudaTensor_nElement(GetCudaState(), tensor);
}
template<>
long Tensor<double, GPU_Cuda>::nElement(const THCudaDoubleTensor *tensor)
{
    return (long)THCudaDoubleTensor_nElement(GetCudaState(), tensor);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long, GPU_Cuda>::narrow(THCudaLongTensor *self, THCudaLongTensor *src, int dimension, long firstIndex, long size)
{
    THCudaLongTensor_narrow(GetCudaState(), self, src, dimension, firstIndex, size);
}
template<>
void Tensor<float, GPU_Cuda>::narrow(THCudaTensor *self, THCudaTensor *src, int dimension, long firstIndex, long size)
{
    THCudaTensor_narrow(GetCudaState(), self, src, dimension, firstIndex, size);
}
template<>
void Tensor<double, GPU_Cuda>::narrow(THCudaDoubleTensor *self, THCudaDoubleTensor *src, int dimension, long firstIndex, long size)
{
    THCudaDoubleTensor_narrow(GetCudaState(), self, src, dimension, firstIndex, size);
}

template<>
void Tensor<long, GPU_Cuda>::select(THCudaLongTensor *self, THCudaLongTensor *src, int dimension, long sliceIndex)
{
    THCudaLongTensor_select(GetCudaState(), self, src, dimension, sliceIndex);
}
template<>
void Tensor<float, GPU_Cuda>::select(THCudaTensor *self, THCudaTensor *src, int dimension, long sliceIndex)
{
    THCudaTensor_select(GetCudaState(), self, src, dimension, sliceIndex);
}
template<>
void Tensor<double, GPU_Cuda>::select(THCudaDoubleTensor *self, THCudaDoubleTensor *src, int dimension, long sliceIndex)
{
    THCudaDoubleTensor_select(GetCudaState(), self, src, dimension, sliceIndex);
}

template<>
void Tensor<long, GPU_Cuda>::transpose(THCudaLongTensor *self, THCudaLongTensor *src, int dimension1, int dimension2)
{
    THCudaLongTensor_transpose(GetCudaState(), self, src, dimension1, dimension2);
}
template<>
void Tensor<float, GPU_Cuda>::transpose(THCudaTensor *self, THCudaTensor *src, int dimension1, int dimension2)
{
    THCudaTensor_transpose(GetCudaState(), self, src, dimension1, dimension2);
}
template<>
void Tensor<double, GPU_Cuda>::transpose(THCudaDoubleTensor *self, THCudaDoubleTensor *src, int dimension1, int dimension2)
{
    THCudaDoubleTensor_transpose(GetCudaState(), self, src, dimension1, dimension2);
}

//////////////////////////////////////////////////////////////////////////

template<>
void Tensor<long, GPU_Cuda>::fill(THCudaLongTensor *r, long val)
{
    return THCudaLongTensor_fill(GetCudaState(), r, val);
}
template<>
void Tensor<float, GPU_Cuda>::fill(THCudaTensor *r, float val)
{
    return THCudaTensor_fill(GetCudaState(), r, val);
}
template<>
void Tensor<double, GPU_Cuda>::fill(THCudaDoubleTensor *r, double val)
{
    return THCudaDoubleTensor_fill(GetCudaState(), r, val);
}

template<>
long Tensor<long, GPU_Cuda>::minall(THCudaLongTensor *r)
{
    return THCudaLongTensor_minall(GetCudaState(), r);
}
template<>
float Tensor<float, GPU_Cuda>::minall(THCudaTensor *r)
{
    return THCudaTensor_minall(GetCudaState(), r);
}
template<>
double Tensor<double, GPU_Cuda>::minall(THCudaDoubleTensor *r)
{
    return THCudaDoubleTensor_minall(GetCudaState(), r);
}

template<>
long Tensor<long, GPU_Cuda>::maxall(THCudaLongTensor *r)
{
    return THCudaLongTensor_maxall(GetCudaState(), r);
}
template<>
float Tensor<float, GPU_Cuda>::maxall(THCudaTensor *r)
{
    return THCudaTensor_maxall(GetCudaState(), r);
}
template<>
double Tensor<double, GPU_Cuda>::maxall(THCudaDoubleTensor *r)
{
    return THCudaDoubleTensor_maxall(GetCudaState(), r);
}

template<>
void Tensor<long, GPU_Cuda>::max(THCudaLongTensor *values, THCudaLongTensor *t, int dimension)
{
    cpptorch::Tensor<long, GPU_Cuda> l(true);
    THCudaLongTensor_max(GetCudaState(), values, l, t, dimension);
}
template<>
void Tensor<float, GPU_Cuda>::max(THCudaTensor *values, THCudaTensor *t, int dimension)
{
    cpptorch::Tensor<long, GPU_Cuda> l(true);
    THCudaTensor_max(GetCudaState(), values, l, t, dimension);
}
template<>
void Tensor<double, GPU_Cuda>::max(THCudaDoubleTensor *values, THCudaDoubleTensor *t, int dimension)
{
    cpptorch::Tensor<long, GPU_Cuda> l(true);
    THCudaDoubleTensor_max(GetCudaState(), values, l, t, dimension);
}

template<>
void Tensor<long, GPU_Cuda>::sum(THCudaLongTensor *values, THCudaLongTensor *t, int dimension)
{
    return THCudaLongTensor_sum(GetCudaState(), values, t, dimension);
}
template<>
void Tensor<float, GPU_Cuda>::sum(THCudaTensor *values, THCudaTensor *t, int dimension)
{
    return THCudaTensor_sum(GetCudaState(), values, t, dimension);
}
template<>
void Tensor<double, GPU_Cuda>::sum(THCudaDoubleTensor *values, THCudaDoubleTensor *t, int dimension)
{
    return THCudaDoubleTensor_sum(GetCudaState(), values, t, dimension);
}

template<>
void Tensor<long, GPU_Cuda>::add(THCudaLongTensor *r, THCudaLongTensor *t, long val)
{
    THCudaLongTensor_add(GetCudaState(), r, t, val);
}
template<>
void Tensor<float, GPU_Cuda>::add(THCudaTensor *r, THCudaTensor *t, float val)
{
    THCudaTensor_add(GetCudaState(), r, t, val);
}
template<>
void Tensor<double, GPU_Cuda>::add(THCudaDoubleTensor *r, THCudaDoubleTensor *t, double val)
{
    THCudaDoubleTensor_add(GetCudaState(), r, t, val);
}

template<>
void Tensor<long, GPU_Cuda>::cadd(THCudaLongTensor *r, THCudaLongTensor *t, long val, THCudaLongTensor *src)
{
    THCudaLongTensor_cadd(GetCudaState(), r, t, val, src);
}
template<>
void Tensor<float, GPU_Cuda>::cadd(THCudaTensor *r, THCudaTensor *t, float val, THCudaTensor *src)
{
    THCudaTensor_cadd(GetCudaState(), r, t, val, src);
}
template<>
void Tensor<double, GPU_Cuda>::cadd(THCudaDoubleTensor *r, THCudaDoubleTensor *t, double val, THCudaDoubleTensor *src)
{
    THCudaDoubleTensor_cadd(GetCudaState(), r, t, val, src);
}

template<>
void Tensor<long, GPU_Cuda>::mul(THCudaLongTensor *r, THCudaLongTensor *t, long val)
{
    THCudaLongTensor_mul(GetCudaState(), r, t, val);
}
template<>
void Tensor<float, GPU_Cuda>::mul(THCudaTensor *r, THCudaTensor *t, float val)
{
    THCudaTensor_mul(GetCudaState(), r, t, val);
}
template<>
void Tensor<double, GPU_Cuda>::mul(THCudaDoubleTensor *r, THCudaDoubleTensor *t, double val)
{
    THCudaDoubleTensor_mul(GetCudaState(), r, t, val);
}

template<>
void Tensor<long, GPU_Cuda>::cmul(THCudaLongTensor *r, THCudaLongTensor *t, THCudaLongTensor *src)
{
    THCudaLongTensor_cmul(GetCudaState(), r, t, src);
}
template<>
void Tensor<float, GPU_Cuda>::cmul(THCudaTensor *r, THCudaTensor *t, THCudaTensor *src)
{
    THCudaTensor_cmul(GetCudaState(), r, t, src);
}
template<>
void Tensor<double, GPU_Cuda>::cmul(THCudaDoubleTensor *r, THCudaDoubleTensor *t, THCudaDoubleTensor *src)
{
    THCudaDoubleTensor_cmul(GetCudaState(), r, t, src);
}

template<>
void Tensor<long, GPU_Cuda>::cdiv(THCudaLongTensor *r, THCudaLongTensor *t, THCudaLongTensor *src)
{
    THCudaLongTensor_cdiv(GetCudaState(), r, t, src);
}
template<>
void Tensor<float, GPU_Cuda>::cdiv(THCudaTensor *r, THCudaTensor *t, THCudaTensor *src)
{
    THCudaTensor_cdiv(GetCudaState(), r, t, src);
}
template<>
void Tensor<double, GPU_Cuda>::cdiv(THCudaDoubleTensor *r, THCudaDoubleTensor *t, THCudaDoubleTensor *src)
{
    THCudaDoubleTensor_cdiv(GetCudaState(), r, t, src);
}

template<>
void Tensor<long, GPU_Cuda>::pow(THCudaLongTensor *r, THCudaLongTensor *t, long val)
{
    assert(0);
}
template<>
void Tensor<float, GPU_Cuda>::pow(THCudaTensor *r, THCudaTensor *t, float val)
{
    THCudaTensor_pow(GetCudaState(), r, t, val);
}
template<>
void Tensor<double, GPU_Cuda>::pow(THCudaDoubleTensor *r, THCudaDoubleTensor *t, double val)
{
    THCudaDoubleTensor_pow(GetCudaState(), r, t, val);
}

template<>
void Tensor<long, GPU_Cuda>::cpow(THCudaLongTensor *r, THCudaLongTensor *t, THCudaLongTensor *src)
{
    THCudaLongTensor_cpow(GetCudaState(), r, t, src);
}
template<>
void Tensor<float, GPU_Cuda>::cpow(THCudaTensor *r, THCudaTensor *t, THCudaTensor *src)
{
    THCudaTensor_cpow(GetCudaState(), r, t, src);
}
template<>
void Tensor<double, GPU_Cuda>::cpow(THCudaDoubleTensor *r, THCudaDoubleTensor *t, THCudaDoubleTensor *src)
{
    THCudaDoubleTensor_cpow(GetCudaState(), r, t, src);
}

template<>
void Tensor<long, GPU_Cuda>::addmv(THCudaLongTensor *r, long beta, THCudaLongTensor *t, long alpha, THCudaLongTensor *mat, THCudaLongTensor *vec)
{
    THCudaLongTensor_addmv(GetCudaState(), r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<float, GPU_Cuda>::addmv(THCudaTensor *r, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec)
{
    THCudaTensor_addmv(GetCudaState(), r, beta, t, alpha, mat, vec);
}
template<>
void Tensor<double, GPU_Cuda>::addmv(THCudaDoubleTensor *r, double beta, THCudaDoubleTensor *t, double alpha, THCudaDoubleTensor *mat, THCudaDoubleTensor *vec)
{
    THCudaDoubleTensor_addmv(GetCudaState(), r, beta, t, alpha, mat, vec);
}

template<>
void Tensor<long, GPU_Cuda>::addmm(THCudaLongTensor *r, long beta, THCudaLongTensor *t, long alpha, THCudaLongTensor *mat1, THCudaLongTensor *mat2)
{
    THCudaLongTensor_addmm(GetCudaState(), r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<float, GPU_Cuda>::addmm(THCudaTensor *r, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat1, THCudaTensor *mat2)
{
    THCudaTensor_addmm(GetCudaState(), r, beta, t, alpha, mat1, mat2);
}
template<>
void Tensor<double, GPU_Cuda>::addmm(THCudaDoubleTensor *r, double beta, THCudaDoubleTensor *t, double alpha, THCudaDoubleTensor *mat1, THCudaDoubleTensor *mat2)
{
    THCudaDoubleTensor_addmm(GetCudaState(), r, beta, t, alpha, mat1, mat2);
}

template<>
void Tensor<long, GPU_Cuda>::addr(THCudaLongTensor *r, long beta, THCudaLongTensor *t, long alpha, THCudaLongTensor *vec1, THCudaLongTensor *vec2)
{
    THCudaLongTensor_addr(GetCudaState(), r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<float, GPU_Cuda>::addr(THCudaTensor *r, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2)
{
    THCudaTensor_addr(GetCudaState(), r, beta, t, alpha, vec1, vec2);
}
template<>
void Tensor<double, GPU_Cuda>::addr(THCudaDoubleTensor *r, double beta, THCudaDoubleTensor *t, double alpha, THCudaDoubleTensor *vec1, THCudaDoubleTensor *vec2)
{
    THCudaDoubleTensor_addr(GetCudaState(), r, beta, t, alpha, vec1, vec2);
}

template<>
void Tensor<long, GPU_Cuda>::abs(THCudaLongTensor *r, THCudaLongTensor *t)
{
    THCudaLongTensor_abs(GetCudaState(), r, t);
}
template<>
void Tensor<float, GPU_Cuda>::abs(THCudaTensor *r, THCudaTensor *t)
{
    THCudaTensor_abs(GetCudaState(), r, t);
}
template<>
void Tensor<double, GPU_Cuda>::abs(THCudaDoubleTensor *r, THCudaDoubleTensor *t)
{
    THCudaDoubleTensor_abs(GetCudaState(), r, t);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

#include <THCUNN/THCUNN.h>

template<>
void NN<float, GPU_Cuda>::BatchNormalization_updateOutput(THCudaTensor *input, THCudaTensor *output,
    THCudaTensor *weight, THCudaTensor *bias, THCudaTensor *running_mean, THCudaTensor *running_var,
    THCudaTensor *save_mean, THCudaTensor *save_std,
    bool train, double momentum, double eps)
{
    THNN_CudaBatchNormalization_updateOutput(GetCudaState(), input, output, weight, bias, running_mean, running_var, save_mean, save_std,
        train, momentum, eps);
}

template<>
void NN<float, GPU_Cuda>::SpatialAveragePooling_updateOutput(THCudaTensor *input, THCudaTensor *output,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)
{
    THNN_CudaSpatialAveragePooling_updateOutput(GetCudaState(), input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}

template<>
void NN<float, GPU_Cuda>::SpatialConvolutionMM_updateOutput(THCudaTensor *input, THCudaTensor *output,
    THCudaTensor *weight, THCudaTensor *bias, THCudaTensor *finput, THCudaTensor *fgradInput,
    int kW, int kH, int dW, int dH, int padW, int padH)
{
    THNN_CudaSpatialConvolutionMM_updateOutput(GetCudaState(), input, output, weight, bias, finput, fgradInput,
        kW, kH, dW, dH, padW, padH);
}

template<>
void NN<float, GPU_Cuda>::SpatialMaxPooling_updateOutput(
    THCudaTensor *input, THCudaTensor *output, THCudaTensor *indices,
    int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
    THNN_CudaSpatialMaxPooling_updateOutput(GetCudaState(), input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}

template<>
void NN<float, GPU_Cuda>::SpatialReflectionPadding_updateOutput(
    THCudaTensor *input, THCudaTensor *output,
    int pad_l, int pad_r, int pad_t, int pad_b)
{
    THNN_CudaSpatialReflectionPadding_updateOutput(GetCudaState(), input, output, pad_l, pad_r, pad_t, pad_b);
}

template<>
void NN<float, GPU_Cuda>::Square_updateOutput(THCudaTensor *input, THCudaTensor *output)
{
    THNN_CudaSquare_updateOutput(GetCudaState(), input, output);
}

template<>
void NN<float, GPU_Cuda>::Sqrt_updateOutput(THCudaTensor *input, THCudaTensor *output, float eps)
{
    THNN_CudaSqrt_updateOutput(GetCudaState(), input, output, eps);
}

template<>
void NN<float, GPU_Cuda>::Threshold_updateOutput(THCudaTensor *input, THCudaTensor *output,
    float threshold, float val, bool inplace)
{
    THNN_CudaThreshold_updateOutput(GetCudaState(), input, output, threshold, val, inplace);
}


}
}
