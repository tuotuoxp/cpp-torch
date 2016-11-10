#include "common.h"
#include <cpptorch_cuda.h>


template <> cpptorch::Tensor<float, GPU_Cuda> read_tensor_template(const cpptorch::object *obj)
{
    return cpptorch::read_cuda_tensor<float>(obj);
}

template <> std::shared_ptr<cpptorch::nn::Layer<float, GPU_Cuda>> read_net_template(const cpptorch::object *obj)
{
    return cpptorch::read_cuda_net<float>(obj);
}


int main(int argc, char *argv[])
{
    int ret = process_args(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    cpptorch::cuda::init();

    ret = test_layer<float, GPU_Cuda>();

    cpptorch::cuda::free();

#ifdef _WIN64
    _CrtDumpMemoryLeaks();
#endif
    return ret;
}
