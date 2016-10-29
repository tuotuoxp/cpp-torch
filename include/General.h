#pragma once


struct THLongStorage;
struct THFloatStorage;
struct THDoubleStorage;
struct THLongTensor;
struct THFloatTensor;
struct THDoubleTensor;

struct THCudaStorage;
struct THCudaLongStorage;
struct THCudaDoubleStorage;
struct THCudaTensor;
struct THCudaLongTensor;
struct THCudaDoubleTensor;


enum GPUFlag
{
    GPU_None = 0,
    GPU_Cuda = 1,
};

namespace cpptorch
{
    template<typename T, GPUFlag F = GPU_None>
    class THTrait
    {
    public:
        struct Tensor {};
        struct Storage {};
    };

    template<> class THTrait<long, GPU_None>
    {
    public:
        using Tensor = THLongTensor;
        using Storage = THLongStorage;
    };

    template<> class THTrait<float, GPU_None>
    {
    public:
        using Tensor = THFloatTensor;
        using Storage = THFloatStorage;
    };

    template<> class THTrait<double, GPU_None>
    {
    public:
        using Tensor = THDoubleTensor;
        using Storage = THDoubleStorage;
    };

    template<> class THTrait<long, GPU_Cuda>
    {
    public:
        using Tensor = THCudaLongTensor;
        using Storage = THCudaLongStorage;
    };

    template<> class THTrait<float, GPU_Cuda>
    {
    public:
        using Tensor = THCudaTensor;
        using Storage = THCudaStorage;
    };

    template<> class THTrait<double, GPU_Cuda>
    {
    public:
        using Tensor = THCudaDoubleTensor;
        using Storage = THCudaDoubleStorage;
    };
}




#ifdef _WIN64
#ifdef API_CPPTORCH_DEF
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif
