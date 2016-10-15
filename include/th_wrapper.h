#pragma once
#include "General.h"


typedef void THNNState;
struct THAllocator;


namespace cpptorch
{
    namespace th
    {
        template<typename T>
        class API Storage
        {
        public:
            // creation methods
            static typename THTrait<T>::Storage* newWithAllocator(THAllocator *allocator, void *allocatorContext);
            static typename THTrait<T>::Storage* newWithDataAndAllocator(T *data, long size,
                THAllocator *allocator, void *allocatorContext);
            static void retain(typename THTrait<T>::Storage *storage);
            static void release(typename THTrait<T>::Storage *storage);

            // getter
            static T* data(typename THTrait<T>::Storage *storage);
            static int size(typename THTrait<T>::Storage *storage);
        };


        template<typename T>
        class API Tensor
        {
        public:
            // creation methods
            static typename THTrait<T>::Tensor* newWithStorage(typename THTrait<T>::Storage *storage, long offset,
                THTrait<long>::Storage *size, THTrait<long>::Storage *stride);
            static void resize(typename THTrait<T>::Tensor *self,
                typename THTrait<long>::Storage *size, typename THTrait<long>::Storage *stride);
            static void resizeAs(typename THTrait<T>::Tensor *self, typename THTrait<T>::Tensor *src);
            static void copy(typename THTrait<T>::Tensor *self, typename THTrait<T>::Tensor *src);
            static void retain(typename THTrait<T>::Tensor *tensor);
            static void release(typename THTrait<T>::Tensor *tensor);

            // direct access methods
            static typename THTrait<T>::Storage* storage(const typename THTrait<T>::Tensor *tensor);
            static long storageOffset(const typename THTrait<T>::Tensor *tensor);
            static int nDimension(const typename THTrait<T>::Tensor *tensor);
            static long size(const typename THTrait<T>::Tensor *tensor, int dim);
            static THTrait<long>::Storage* size(const typename THTrait<T>::Tensor *tensor);
            static THTrait<long>::Storage* stride(const typename THTrait<T>::Tensor *tensor);
            static T* data(const typename THTrait<T>::Tensor *tensor);

            // calculative access methods
            static int isContiguous(const typename THTrait<T>::Tensor *tensor);
            static int nElement(const typename THTrait<T>::Tensor *tensor);

            // special access methods (shares the same storage)
            static void narrow(typename THTrait<T>::Tensor *self, typename THTrait<T>::Tensor *src, int dimension, long firstIndex, long size);
            static void select(typename THTrait<T>::Tensor *self, typename THTrait<T>::Tensor *src, int dimension, long sliceIndex);
            static void transpose(typename THTrait<T>::Tensor *self, typename THTrait<T>::Tensor *src, int dimension1, int dimension2);

            // maths
            static void fill(typename THTrait<T>::Tensor *self, T val);
            static T minall(typename THTrait<T>::Tensor *self);
            static T maxall(typename THTrait<T>::Tensor *self);
            static void max(typename THTrait<T>::Tensor *values, typename THTrait<T>::Tensor *t, int dimension);
            static void sum(typename THTrait<T>::Tensor *values, typename THTrait<T>::Tensor *t, int dimension);
            // r = t + val
            static void add(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t,
                T val);
            // r = t + val * src
            static void cadd(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t,
                T val, typename THTrait<T>::Tensor *src);
            // r = t * val
            static void mul(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t,
                T val);
            // r = t * src
            static void cmul(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t, typename THTrait<T>::Tensor *src);
            // r = t / src
            static void cdiv(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t, typename THTrait<T>::Tensor *src);
            // r = t ^ val
            static void pow(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t,
                T val);
            // r = pow(t, src)
            static void cpow(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t, typename THTrait<T>::Tensor *src);
            // r = beta * t + alpha * (mat * vec)
            static void addmv(typename THTrait<T>::Tensor *r, T beta,
                typename THTrait<T>::Tensor *t, T alpha,
                typename THTrait<T>::Tensor *mat, typename THTrait<T>::Tensor *vec);
            // r = beta * t + alpha * (mat1 * mat1)
            static void addmm(typename THTrait<T>::Tensor *r, T beta,
                typename THTrait<T>::Tensor *t, T alpha,
                typename THTrait<T>::Tensor *mat1, typename THTrait<T>::Tensor *mat2);
            // r = beta * t + alpha * (vec1 x vec2)
            static void addr(typename THTrait<T>::Tensor *r, T beta,
                typename THTrait<T>::Tensor *t, T alpha,
                typename THTrait<T>::Tensor *vec1, typename THTrait<T>::Tensor *vec2);
            // r = abs(t)
            static void abs(typename THTrait<T>::Tensor *r, typename THTrait<T>::Tensor *t);
        };


        template<typename T>
        class API NN
        {
        public:
            static void BatchNormalization_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output,
                typename THTrait<T>::Tensor *weight, typename THTrait<T>::Tensor *bias,
                typename THTrait<T>::Tensor *running_mean, typename THTrait<T>::Tensor *running_var,
                typename THTrait<T>::Tensor *save_mean, typename THTrait<T>::Tensor *save_std,
                bool train, double momentum, double eps);

            static void SpatialAveragePooling_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output,
                int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);

            static void SpatialConvolutionMM_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output,
                typename THTrait<T>::Tensor *weight, typename THTrait<T>::Tensor *bias,
                typename THTrait<T>::Tensor *finput, typename THTrait<T>::Tensor *fgradInput,
                int kW, int kH, int dW, int dH, int padW, int padH);

            static void SpatialMaxPooling_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output,
                typename THTrait<T>::Tensor *indices,
                int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);

            static void SpatialReflectionPadding_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output,
                int pad_l, int pad_r, int pad_t, int pad_b);

            static void Sqrt_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output,
                T eps);

            static void Square_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output);

            static void Threshold_updateOutput(THNNState *state,
                typename THTrait<T>::Tensor *input, typename THTrait<T>::Tensor *output,
                T threshold, T val,
                bool inplace);
        };
    }
}
