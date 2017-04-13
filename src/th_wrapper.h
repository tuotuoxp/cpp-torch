#pragma once
#include "../include/General.h"


typedef void THNNState;
struct THAllocator;


namespace cpptorch
{
    namespace th
    {
        template<typename T, GPUFlag F>
        class API Storage
        {
        public:
            // creation methods
            static typename THTrait<T, F>::Storage* newWithData(const T *ptr_src, long count, bool take_ownership_of_data);
            static void retain(typename THTrait<T, F>::Storage *storage);
            static void release(typename THTrait<T, F>::Storage *storage);

            // getter
            static T* data(typename THTrait<T, F>::Storage *storage);
            static T data_by_index(const typename THTrait<T, F>::Storage *storage, long index);
            static long size(const typename THTrait<T, F>::Storage *storage);
        };


        template<typename T, GPUFlag F>
        class API Tensor
        {
        public:
            // creation methods
            static typename THTrait<T, F>::Tensor* newWithStorage(typename THTrait<T, F>::Storage *storage, long offset,
                int dim, const long *size, const long *stride);
            static void resize(typename THTrait<T, F>::Tensor *self,
                typename THTrait<long, GPU_None>::Storage *size, typename THTrait<long, GPU_None>::Storage *stride);
            static void resizeAs(typename THTrait<T, F>::Tensor *self, typename THTrait<T, F>::Tensor *src);
            static void copy(typename THTrait<T, F>::Tensor *self, typename THTrait<T, F>::Tensor *src);
            static void retain(typename THTrait<T, F>::Tensor *tensor);
            static void release(typename THTrait<T, F>::Tensor *tensor);

            // direct access methods
            static typename THTrait<T, F>::Storage* storage(const typename THTrait<T, F>::Tensor *tensor);
            static long storageOffset(const typename THTrait<T, F>::Tensor *tensor);
            static int nDimension(const typename THTrait<T, F>::Tensor *tensor);
            static long size(const typename THTrait<T, F>::Tensor *tensor, int dim);
            static typename THTrait<long, GPU_None>::Storage* size(const typename THTrait<T, F>::Tensor *tensor);
            static typename THTrait<long, GPU_None>::Storage* stride(const typename THTrait<T, F>::Tensor *tensor);
            static T* data(const typename THTrait<T, F>::Tensor *tensor);

            // calculative access methods
            static int isContiguous(const typename THTrait<T, F>::Tensor *tensor);
            static int isSameSizeAs(const typename THTrait<T, F>::Tensor *self, const typename THTrait<T, F>::Tensor *src);
            static long nElement(const typename THTrait<T, F>::Tensor *tensor);

            // special access methods (shares the same storage)
            static void narrow(typename THTrait<T, F>::Tensor *self, typename THTrait<T, F>::Tensor *src, int dimension, long firstIndex, long size);
            static void select(typename THTrait<T, F>::Tensor *self, typename THTrait<T, F>::Tensor *src, int dimension, long sliceIndex);
            static void transpose(typename THTrait<T, F>::Tensor *self, typename THTrait<T, F>::Tensor *src, int dimension1, int dimension2);

            // maths
            static void fill(typename THTrait<T, F>::Tensor *self, T val);
            static T minall(typename THTrait<T, F>::Tensor *self);
            static T maxall(typename THTrait<T, F>::Tensor *self);
            static void max(typename THTrait<T, F>::Tensor *values, typename THTrait<T, F>::Tensor *t, int dimension);
            static void sum(typename THTrait<T, F>::Tensor *values, typename THTrait<T, F>::Tensor *t, int dimension);
            // r = t + val
            static void add(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t,
                T val);
            // r = t + val * src
            static void cadd(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t,
                T val, typename THTrait<T, F>::Tensor *src);
            // r = t * val
            static void mul(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t,
                T val);
            // r = t * src
            static void cmul(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t, typename THTrait<T, F>::Tensor *src);
            // r = t / src
            static void cdiv(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t, typename THTrait<T, F>::Tensor *src);
            // r = t ^ val
            static void pow(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t,
                T val);
            // r = pow(t, src)
            static void cpow(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t, typename THTrait<T, F>::Tensor *src);
            // r = beta * t + alpha * (mat * vec)
            static void addmv(typename THTrait<T, F>::Tensor *r, T beta,
                typename THTrait<T, F>::Tensor *t, T alpha,
                typename THTrait<T, F>::Tensor *mat, typename THTrait<T, F>::Tensor *vec);
            // r = beta * t + alpha * (mat1 * mat1)
            static void addmm(typename THTrait<T, F>::Tensor *r, T beta,
                typename THTrait<T, F>::Tensor *t, T alpha,
                typename THTrait<T, F>::Tensor *mat1, typename THTrait<T, F>::Tensor *mat2);
            // r = beta * t + alpha * (vec1 x vec2)
            static void addr(typename THTrait<T, F>::Tensor *r, T beta,
                typename THTrait<T, F>::Tensor *t, T alpha,
                typename THTrait<T, F>::Tensor *vec1, typename THTrait<T, F>::Tensor *vec2);
            // r = abs(t)
            static void abs(typename THTrait<T, F>::Tensor *r, typename THTrait<T, F>::Tensor *t);
        };


        template<typename T, GPUFlag F>
        class NN
        {
        public:
            static void BatchNormalization_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output,
                typename THTrait<T, F>::Tensor *weight, typename THTrait<T, F>::Tensor *bias,
                typename THTrait<T, F>::Tensor *running_mean, typename THTrait<T, F>::Tensor *running_var,
                typename THTrait<T, F>::Tensor *save_mean, typename THTrait<T, F>::Tensor *save_std,
                bool train, double momentum, double eps);

            static void SpatialAveragePooling_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output,
                int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);

            static void SpatialConvolutionMM_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output,
                typename THTrait<T, F>::Tensor *weight, typename THTrait<T, F>::Tensor *bias,
                typename THTrait<T, F>::Tensor *finput, typename THTrait<T, F>::Tensor *fgradInput,
                int kW, int kH, int dW, int dH, int padW, int padH);

            static void SpatialMaxPooling_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output,
                typename THTrait<long, F>::Tensor *indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);

            static void SpatialReflectionPadding_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output,
                int pad_l, int pad_r, int pad_t, int pad_b);

            static void Sqrt_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output,
                T eps);

            static void Square_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output);

            static void Threshold_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output,
                T threshold, T val,
                bool inplace);
            static void LogSoftMax_updateOutput(
                typename THTrait<T, F>::Tensor *input, typename THTrait<T, F>::Tensor *output);
        };
    }
}
