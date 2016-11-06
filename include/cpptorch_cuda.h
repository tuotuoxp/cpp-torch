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

    API CudaTensor cpu2cuda(const Tensor<float> &t);
    API Tensor<float> cuda2cpu(const CudaTensor &t);


    namespace cuda
    {
        API void init();
        API void free();
    }
}
