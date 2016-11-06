#pragma once
#include "../include/General.h"


namespace cpptorch
{
    namespace th
    {
        template<typename T>
        void copy_cpu2cuda(typename THTrait<T, GPU_Cuda>::Tensor *self, typename THTrait<T>::Tensor *src);

        template<typename T>
        void copy_cuda2cpu(typename THTrait<T>::Tensor *self, typename THTrait<T, GPU_Cuda>::Tensor *src);
    }
}

