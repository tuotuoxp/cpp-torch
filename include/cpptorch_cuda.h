#pragma once
#include "cpptorch.h"


struct THCState;


namespace cpptorch
{
    typedef Tensor<float, true> CudaTensor;
    namespace nn
    {
        typedef Layer<float, true> CudaLayer;
    }

    API CudaTensor read_cuda_tensor(const object *obj);
    API std::shared_ptr<nn::CudaLayer> read_cuda_net(const object *obj);

    namespace cuda
    {
        API void init();
        API void free();
    }
}
