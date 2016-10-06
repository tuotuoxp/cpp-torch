#pragma once
#include <TH/TH.h>
#include <THNN/THNN.h>


class THWrapper
{
public:
    template<class TStorage>
    class Storage
    {
    public:
        // creation methods
        static typename TStorage::THStorage* newWithDataAndAllocator(typename TStorage::StorageBase *data, long size,
            THAllocator *allocator, void *allocatorContext);
        static void retain(typename TStorage::THStorage *storage);
        static void release(typename TStorage::THStorage *storage);

        // getter
        static typename TStorage::StorageBase* data(typename TStorage::THStorage *storage);
        static int size(typename TStorage::THStorage *storage);
    };


    template<class TTensor>
    class Tensor
    {
    public:
        // creation methods
        static typename TTensor::THTensor* create();
        static typename TTensor::THTensor* newWithStorage(typename TTensor::Storage::THStorage *storage, long offset,
            typename TTensor::SizeStorage::THStorage *size, typename TTensor::SizeStorage::THStorage *stride);
        static void resize(typename TTensor::THTensor *self,
            typename TTensor::SizeStorage::THStorage *size, typename TTensor::SizeStorage::THStorage *stride);
        static void resizeAs(typename TTensor::THTensor *self, typename TTensor::THTensor *src);
        static void copy(typename TTensor::THTensor *self, typename TTensor::THTensor *src);
        static void retain(typename TTensor::THTensor *tensor);
        static void release(typename TTensor::THTensor *tensor);

        // direct access methods
        static typename TTensor::Storage::THStorage* storage(const typename TTensor::THTensor *tensor);
        static long storageOffset(const typename TTensor::THTensor *tensor);
        static int nDimension(const typename TTensor::THTensor *tensor);
        static long size(const typename TTensor::THTensor *tensor, int dim);
        static typename TTensor::SizeStorage::THStorage* size(const typename TTensor::THTensor *tensor);
        static typename TTensor::SizeStorage::THStorage* stride(const typename TTensor::THTensor *tensor);
        static typename TTensor::Storage::StorageBase* data(const typename TTensor::THTensor *tensor);

        // calculative access methods
        static int isContiguous(const typename TTensor::THTensor *tensor);
        static int nElement(const typename TTensor::THTensor *tensor);

        // special access methods (shares the same storage)
        static void narrow(typename TTensor::THTensor *self, typename TTensor::THTensor *src, int dimension, long firstIndex, long size);
        static void select(typename TTensor::THTensor *self, typename TTensor::THTensor *src, int dimension, long sliceIndex);
        static void transpose(typename TTensor::THTensor *self, typename TTensor::THTensor *src, int dimension1, int dimension2);

        // maths
        static void fill(typename TTensor::THTensor *self, typename TTensor::Storage::StorageBase val);
        static typename TTensor::Storage::StorageBase minall(typename TTensor::THTensor *self);
        static typename TTensor::Storage::StorageBase maxall(typename TTensor::THTensor *self);
        static void max(typename TTensor::THTensor *values, typename TTensor::THTensor *t, int dimension);
        static void sum(typename TTensor::THTensor *values, typename TTensor::THTensor *t, int dimension);
        // r = t + val
        static void add(typename TTensor::THTensor *r, typename TTensor::THTensor *t,
                        typename TTensor::Storage::StorageBase val);
        // r = t + val * src
        static void cadd(typename TTensor::THTensor *r, typename TTensor::THTensor *t,
                         typename TTensor::Storage::StorageBase val, typename TTensor::THTensor *src);
        // r = t * val
        static void mul(typename TTensor::THTensor *r, typename TTensor::THTensor *t,
            typename TTensor::Storage::StorageBase val);
        // r = t * src
        static void cmul(typename TTensor::THTensor *r, typename TTensor::THTensor *t, typename TTensor::THTensor *src);
        // r = t / src
        static void cdiv(typename TTensor::THTensor *r, typename TTensor::THTensor *t, typename TTensor::THTensor *src);
        // r = t ^ val
        static void pow(typename TTensor::THTensor *r, typename TTensor::THTensor *t,
                        typename TTensor::Storage::StorageBase val);
        // r = pow(t, src)
        static void cpow(typename TTensor::THTensor *r, typename TTensor::THTensor *t, typename TTensor::THTensor *src);
        // r = beta * t + alpha * (mat * vec)
        static void addmv(typename TTensor::THTensor *r, typename TTensor::Storage::StorageBase beta,
            typename TTensor::THTensor *t, typename TTensor::Storage::StorageBase alpha,
            typename TTensor::THTensor *mat, typename TTensor::THTensor *vec);
        // r = beta * t + alpha * (mat1 * mat1)
        static void addmm(typename TTensor::THTensor *r, typename TTensor::Storage::StorageBase beta,
            typename TTensor::THTensor *t, typename TTensor::Storage::StorageBase alpha,
            typename TTensor::THTensor *mat1, typename TTensor::THTensor *mat2);
        // r = beta * t + alpha * (vec1 x vec2)
        static void addr(typename TTensor::THTensor *r, typename TTensor::Storage::StorageBase beta,
            typename TTensor::THTensor *t, typename TTensor::Storage::StorageBase alpha,
            typename TTensor::THTensor *vec1, typename TTensor::THTensor *vec2);
        // r = abs(t)
        static void abs(typename TTensor::THTensor *r, typename TTensor::THTensor *t);
    };


    template<class TTensor>
    class NN
    {
    public:
        static void SpatialConvolutionMM_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output,
            typename TTensor::THTensor *weight, typename TTensor::THTensor *bias,
            typename TTensor::THTensor *finput, typename TTensor::THTensor *fgradInput,
            int kW, int kH, int dW, int dH, int padW, int padH);

        static void BatchNormalization_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output,
            typename TTensor::THTensor *weight, typename TTensor::THTensor *bias,
            typename TTensor::THTensor *running_mean, typename TTensor::THTensor *running_var,
            typename TTensor::THTensor *save_mean, typename TTensor::THTensor *save_std,
            bool train, double momentum, double eps);

        static void Threshold_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output,
            typename TTensor::Storage::StorageBase threshold, typename TTensor::Storage::StorageBase val,
            bool inplace);

        static void SpatialMaxPooling_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output,
            typename TTensor::THTensor *indices,
            int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);

        static void Square_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output);

        static void SpatialAveragePooling_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output
        );

        static void SpatialAveragePooling_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output,
            int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);

        static void Sqrt_updateOutput(THNNState *state,
            typename TTensor::THTensor *input, typename TTensor::THTensor *output,
            typename TTensor::Storage::StorageBase eps);
    };
};
