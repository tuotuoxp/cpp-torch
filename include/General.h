#pragma once


struct THLongStorage;
struct THFloatStorage;
struct THLongTensor;
struct THFloatTensor;


/**
 * TensorBase:  TensorBaseFloat
 */

namespace nn
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

typedef nn::StorageTrait<THLongStorage, long> StorageLong;
typedef nn::StorageTrait<THFloatStorage, float> StorageFloat;
typedef nn::TensorTrait<THLongTensor, StorageLong, StorageLong> TensorLong;
typedef nn::TensorTrait<THFloatTensor, StorageFloat, StorageLong> TensorFloat;


#ifdef _WIN64
#ifdef API_CPPTORCH_DEF
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif
