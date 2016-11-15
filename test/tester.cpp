#include "common.h"


template <> cpptorch::Tensor<float, GPU_None> read_tensor_template(const cpptorch::object *obj)
{
    return cpptorch::read_tensor<float>(obj);
}

template <> std::shared_ptr<cpptorch::nn::Layer<float, GPU_None>> read_net_template(const cpptorch::object *obj)
{
    return cpptorch::read_net<float>(obj);
}



int main(int argc, char *argv[])
{
    int ret = process_args(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    if (use_allocator)
    {
        cpptorch::allocator::init();
    }

    ret = test_layer<float, GPU_None>();

    if (use_allocator)
    {
        cpptorch::allocator::cleanup();
    }

#ifdef _WIN64
    _CrtDumpMemoryLeaks();
#endif
    return ret;
}
