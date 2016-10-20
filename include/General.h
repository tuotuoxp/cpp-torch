#pragma once


struct THLongStorage;
struct THFloatStorage;
struct THDoubleStorage;
struct THLongTensor;
struct THFloatTensor;
struct THDoubleTensor;


namespace cpptorch
{

    template<typename T>
    class THTrait
    {
    public:
        struct Tensor {};
        struct Storage {};
    };

    template<> class THTrait<long>
    {
    public:
        using Tensor = THLongTensor;
        using Storage = THLongStorage;
    };

    template<> class THTrait<float>
    {
    public:
        using Tensor = THFloatTensor;
        using Storage = THFloatStorage;
    };

    template<> class THTrait<double>
    {
    public:
        using Tensor = THDoubleTensor;
        using Storage = THDoubleStorage;
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
