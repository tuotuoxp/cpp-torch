#pragma once
#include "cpptorch.h"


struct THCState;


namespace cpptorch
{
    typedef Tensor<float, GPU_Cuda> CudaTensor;
    namespace nn
    {
        typedef Layer<float, GPU_Cuda> CudaLayer;
    }

    API CudaTensor read_cuda_tensor(const object *obj);
    API std::shared_ptr<nn::CudaLayer> read_cuda_net(const object *obj);

    namespace cuda
    {
        API void init();
        API void free();
    }
}
