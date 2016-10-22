#pragma once


struct THLongStorage;
struct THFloatStorage;
struct THDoubleStorage;
struct THLongTensor;
struct THFloatTensor;
struct THDoubleTensor;

struct THCLongStorage;
struct THCFloatStorage;
struct THCDoubleStorage;
struct THCLongTensor;
struct THCFloatTensor;
struct THCDoubleTensor;


namespace cpptorch
{
    template<typename T, bool C = false>
    class THTrait
    {
    public:
        struct Tensor {};
        struct Storage {};
    };

    template<> class THTrait<long, false>
    {
    public:
        using Tensor = THLongTensor;
        using Storage = THLongStorage;
    };

    template<> class THTrait<float, false>
    {
    public:
        using Tensor = THFloatTensor;
        using Storage = THFloatStorage;
    };

    template<> class THTrait<double, false>
    {
    public:
        using Tensor = THDoubleTensor;
        using Storage = THDoubleStorage;
    };

    template<> class THTrait<long, true>
    {
    public:
        using Tensor = THCLongTensor;
        using Storage = THCLongStorage;
    };

    template<> class THTrait<float, true>
    {
    public:
        using Tensor = THCFloatTensor;
        using Storage = THCFloatStorage;
    };

    template<> class THTrait<double, true>
    {
    public:
        using Tensor = THCDoubleTensor;
        using Storage = THCDoubleStorage;
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
