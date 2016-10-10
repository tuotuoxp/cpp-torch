#pragma once
#include "General.h"


typedef void THNNState;
struct THAllocator;


namespace cpptorch
{
    namespace th
    {
        template<class TStorage>
        class Storage
        {
        public:
            // creation methods
            static typename TStorage::TH* newWithDataAndAllocator(typename TStorage::Base *data, long size,
                THAllocator *allocator, void *allocatorContext);
            static void retain(typename TStorage::TH *storage);
            static void release(typename TStorage::TH *storage);

            // getter
            static typename TStorage::Base* data(typename TStorage::TH *storage);
            static int size(typename TStorage::TH *storage);
        };


        template<class TTensor>
        class API Tensor
        {
        public:
            // creation methods
            static typename TTensor::TH* create();
            static typename TTensor::TH* newWithStorage(typename TTensor::Storage::TH *storage, long offset,
                typename TTensor::SizeStorage::TH *size, typename TTensor::SizeStorage::TH *stride);
            static void resize(typename TTensor::TH *self,
                typename TTensor::SizeStorage::TH *size, typename TTensor::SizeStorage::TH *stride);
            static void resizeAs(typename TTensor::TH *self, typename TTensor::TH *src);
            static void copy(typename TTensor::TH *self, typename TTensor::TH *src);
            static void retain(typename TTensor::TH *tensor);
            static void release(typename TTensor::TH *tensor);

            // direct access methods
            static typename TTensor::Storage::TH* storage(const typename TTensor::TH *tensor);
            static long storageOffset(const typename TTensor::TH *tensor);
            static int nDimension(const typename TTensor::TH *tensor);
            static long size(const typename TTensor::TH *tensor, int dim);
            static typename TTensor::SizeStorage::TH* size(const typename TTensor::TH *tensor);
            static typename TTensor::SizeStorage::TH* stride(const typename TTensor::TH *tensor);
            static typename TTensor::Storage::Base* data(const typename TTensor::TH *tensor);

            // calculative access methods
            static int isContiguous(const typename TTensor::TH *tensor);
            static int nElement(const typename TTensor::TH *tensor);

            // special access methods (shares the same storage)
            static void narrow(typename TTensor::TH *self, typename TTensor::TH *src, int dimension, long firstIndex, long size);
            static void select(typename TTensor::TH *self, typename TTensor::TH *src, int dimension, long sliceIndex);
            static void transpose(typename TTensor::TH *self, typename TTensor::TH *src, int dimension1, int dimension2);

            // maths
            static void fill(typename TTensor::TH *self, typename TTensor::Storage::Base val);
            static typename TTensor::Storage::Base minall(typename TTensor::TH *self);
            static typename TTensor::Storage::Base maxall(typename TTensor::TH *self);
            static void max(typename TTensor::TH *values, typename TTensor::TH *t, int dimension);
            static void sum(typename TTensor::TH *values, typename TTensor::TH *t, int dimension);
            // r = t + val
            static void add(typename TTensor::TH *r, typename TTensor::TH *t,
                typename TTensor::Storage::Base val);
            // r = t + val * src
            static void cadd(typename TTensor::TH *r, typename TTensor::TH *t,
                typename TTensor::Storage::Base val, typename TTensor::TH *src);
            // r = t * val
            static void mul(typename TTensor::TH *r, typename TTensor::TH *t,
                typename TTensor::Storage::Base val);
            // r = t * src
            static void cmul(typename TTensor::TH *r, typename TTensor::TH *t, typename TTensor::TH *src);
            // r = t / src
            static void cdiv(typename TTensor::TH *r, typename TTensor::TH *t, typename TTensor::TH *src);
            // r = t ^ val
            static void pow(typename TTensor::TH *r, typename TTensor::TH *t,
                typename TTensor::Storage::Base val);
            // r = pow(t, src)
            static void cpow(typename TTensor::TH *r, typename TTensor::TH *t, typename TTensor::TH *src);
            // r = beta * t + alpha * (mat * vec)
            static void addmv(typename TTensor::TH *r, typename TTensor::Storage::Base beta,
                typename TTensor::TH *t, typename TTensor::Storage::Base alpha,
                typename TTensor::TH *mat, typename TTensor::TH *vec);
            // r = beta * t + alpha * (mat1 * mat1)
            static void addmm(typename TTensor::TH *r, typename TTensor::Storage::Base beta,
                typename TTensor::TH *t, typename TTensor::Storage::Base alpha,
                typename TTensor::TH *mat1, typename TTensor::TH *mat2);
            // r = beta * t + alpha * (vec1 x vec2)
            static void addr(typename TTensor::TH *r, typename TTensor::Storage::Base beta,
                typename TTensor::TH *t, typename TTensor::Storage::Base alpha,
                typename TTensor::TH *vec1, typename TTensor::TH *vec2);
            // r = abs(t)
            static void abs(typename TTensor::TH *r, typename TTensor::TH *t);
        };


        template<class TTensor>
        class API NN
        {
        public:
            static void BatchNormalization_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output,
                typename TTensor::TH *weight, typename TTensor::TH *bias,
                typename TTensor::TH *running_mean, typename TTensor::TH *running_var,
                typename TTensor::TH *save_mean, typename TTensor::TH *save_std,
                bool train, double momentum, double eps);

            static void SpatialAveragePooling_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output,
                int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);

            static void SpatialConvolutionMM_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output,
                typename TTensor::TH *weight, typename TTensor::TH *bias,
                typename TTensor::TH *finput, typename TTensor::TH *fgradInput,
                int kW, int kH, int dW, int dH, int padW, int padH);

            static void SpatialMaxPooling_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output,
                typename TTensor::TH *indices,
                int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);

            static void SpatialReflectionPadding_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output,
                int pad_l, int pad_r, int pad_t, int pad_b);

            static void Sqrt_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output,
                typename TTensor::Storage::Base eps);

            static void Square_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output);

            static void Threshold_updateOutput(THNNState *state,
                typename TTensor::TH *input, typename TTensor::TH *output,
                typename TTensor::Storage::Base threshold, typename TTensor::Storage::Base val,
                bool inplace);
        };
    }
}
