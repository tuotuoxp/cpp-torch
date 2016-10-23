#pragma once
#include "cpptorch.h"


namespace cpptorch
{
    template<typename T>
    API Tensor<T, true> read_cuda_tensor(const object *obj);
    template<typename T>
    API std::shared_ptr<nn::Layer<T, true>> read_cuda_net(const object *obj);
}
