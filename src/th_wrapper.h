#pragma once
#include "../include/General.h"


typedef void THNNState;
struct THAllocator;


namespace cpptorch
{
    namespace th
    {
        template<typename T, bool C>
        class API Storage
        {
        public:
            // creation methods
            static typename THTrait<T,C>::Storage* newWithAllocator(THAllocator *allocator, void *allocatorContext);
            static typename THTrait<T,C>::Storage* newWithDataAndAllocator(T *data, long size,
                THAllocator *allocator, void *allocatorContext);
            static void retain(typename THTrait<T,C>::Storage *storage);
            static void release(typename THTrait<T,C>::Storage *storage);

            // getter
            static T* data(typename THTrait<T,C>::Storage *storage);
            static long size(typename THTrait<T,C>::Storage *storage);
        };


        template<typename T, bool C>
        class API Tensor
        {
        public:
            // creation methods
            static typename THTrait<T,C>::Tensor* newWithStorage(typename THTrait<T,C>::Storage *storage, long offset,
                int dim, const long *size, const long *stride);
            static void resize(typename THTrait<T,C>::Tensor *self,
                typename THTrait<long, false>::Storage *size, typename THTrait<long, false>::Storage *stride);
            static void resizeAs(typename THTrait<T,C>::Tensor *self, typename THTrait<T,C>::Tensor *src);
            static void copy(typename THTrait<T,C>::Tensor *self, typename THTrait<T,C>::Tensor *src);
            static void retain(typename THTrait<T,C>::Tensor *tensor);
            static void release(typename THTrait<T,C>::Tensor *tensor);

            // direct access methods
            static typename THTrait<T,C>::Storage* storage(const typename THTrait<T,C>::Tensor *tensor);
            static long storageOffset(const typename THTrait<T,C>::Tensor *tensor);
            static int nDimension(const typename THTrait<T,C>::Tensor *tensor);
            static long size(const typename THTrait<T,C>::Tensor *tensor, int dim);
            static typename THTrait<long, false>::Storage* size(const typename THTrait<T,C>::Tensor *tensor);
            static typename THTrait<long, false>::Storage* stride(const typename THTrait<T,C>::Tensor *tensor);
            static T* data(const typename THTrait<T,C>::Tensor *tensor);

            // calculative access methods
            static int isContiguous(const typename THTrait<T,C>::Tensor *tensor);
            static long nElement(const typename THTrait<T,C>::Tensor *tensor);

            // special access methods (shares the same storage)
            static void narrow(typename THTrait<T,C>::Tensor *self, typename THTrait<T,C>::Tensor *src, int dimension, long firstIndex, long size);
            static void select(typename THTrait<T,C>::Tensor *self, typename THTrait<T,C>::Tensor *src, int dimension, long sliceIndex);
            static void transpose(typename THTrait<T,C>::Tensor *self, typename THTrait<T,C>::Tensor *src, int dimension1, int dimension2);

            // maths
            static void fill(typename THTrait<T,C>::Tensor *self, T val);
            static T minall(typename THTrait<T,C>::Tensor *self);
            static T maxall(typename THTrait<T,C>::Tensor *self);
            static void max(typename THTrait<T,C>::Tensor *values, typename THTrait<T,C>::Tensor *t, int dimension);
            static void sum(typename THTrait<T,C>::Tensor *values, typename THTrait<T,C>::Tensor *t, int dimension);
            // r = t + val
            static void add(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t,
                T val);
            // r = t + val * src
            static void cadd(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t,
                T val, typename THTrait<T,C>::Tensor *src);
            // r = t * val
            static void mul(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t,
                T val);
            // r = t * src
            static void cmul(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t, typename THTrait<T,C>::Tensor *src);
            // r = t / src
            static void cdiv(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t, typename THTrait<T,C>::Tensor *src);
            // r = t ^ val
            static void pow(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t,
                T val);
            // r = pow(t, src)
            static void cpow(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t, typename THTrait<T,C>::Tensor *src);
            // r = beta * t + alpha * (mat * vec)
            static void addmv(typename THTrait<T,C>::Tensor *r, T beta,
                typename THTrait<T,C>::Tensor *t, T alpha,
                typename THTrait<T,C>::Tensor *mat, typename THTrait<T,C>::Tensor *vec);
            // r = beta * t + alpha * (mat1 * mat1)
            static void addmm(typename THTrait<T,C>::Tensor *r, T beta,
                typename THTrait<T,C>::Tensor *t, T alpha,
                typename THTrait<T,C>::Tensor *mat1, typename THTrait<T,C>::Tensor *mat2);
            // r = beta * t + alpha * (vec1 x vec2)
            static void addr(typename THTrait<T,C>::Tensor *r, T beta,
                typename THTrait<T,C>::Tensor *t, T alpha,
                typename THTrait<T,C>::Tensor *vec1, typename THTrait<T,C>::Tensor *vec2);
            // r = abs(t)
            static void abs(typename THTrait<T,C>::Tensor *r, typename THTrait<T,C>::Tensor *t);
        };


        template<typename T, bool C>
        class NN
        {
        public:
            static void BatchNormalization_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output,
                typename THTrait<T,C>::Tensor *weight, typename THTrait<T,C>::Tensor *bias,
                typename THTrait<T,C>::Tensor *running_mean, typename THTrait<T,C>::Tensor *running_var,
                typename THTrait<T,C>::Tensor *save_mean, typename THTrait<T,C>::Tensor *save_std,
                bool train, double momentum, double eps);

            static void SpatialAveragePooling_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output,
                int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);

            static void SpatialConvolutionMM_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output,
                typename THTrait<T,C>::Tensor *weight, typename THTrait<T,C>::Tensor *bias,
                typename THTrait<T,C>::Tensor *finput, typename THTrait<T,C>::Tensor *fgradInput,
                int kW, int kH, int dW, int dH, int padW, int padH);

            static void SpatialMaxPooling_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output,
                typename THTrait<T,C>::Tensor *indices,
                int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);

            static void SpatialReflectionPadding_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output,
                int pad_l, int pad_r, int pad_t, int pad_b);

            static void Sqrt_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output,
                T eps);

            static void Square_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output);

            static void Threshold_updateOutput(void *state,
                typename THTrait<T,C>::Tensor *input, typename THTrait<T,C>::Tensor *output,
                T threshold, T val,
                bool inplace);
        };
    }
}
