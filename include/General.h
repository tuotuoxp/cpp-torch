#pragma once


struct THLongStorage;
struct THFloatStorage;
struct THDoubleStorage;
struct THLongTensor;
struct THFloatTensor;
struct THDoubleTensor;


namespace cpptorch
{
    template <class TTHStorage, typename TStorageBase>
    class StorageTrait
    {
    public:
        typedef TTHStorage TH;
        typedef TStorageBase Base;
    };

    template <class TTHTensor, class TStorage, class TSizeStorage>
    class TensorTrait
    {
    public:
        typedef TTHTensor TH;
        typedef TStorage Storage;
        typedef TSizeStorage SizeStorage;
    };
}

typedef cpptorch::StorageTrait<THLongStorage, long> StorageLong;
typedef cpptorch::StorageTrait<THFloatStorage, float> StorageFloat;
typedef cpptorch::StorageTrait<THDoubleStorage, double> StorageDouble;
typedef cpptorch::TensorTrait<THLongTensor, StorageLong, StorageLong> TensorLong;
typedef cpptorch::TensorTrait<THFloatTensor, StorageFloat, StorageLong> TensorFloat;
typedef cpptorch::TensorTrait<THDoubleTensor, StorageDouble, StorageLong> TensorDouble;


#ifdef _WIN64
#ifdef API_CPPTORCH_DEF
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif
