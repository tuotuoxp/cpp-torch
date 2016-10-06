#pragma once
#include <TH/TH.h>


/**
 * TensorBase:  TensorBaseFloat
 */

namespace nn
{
    template <class TTHStorage, typename TStorageBase>
    class StorageTrait
    {
    public:
        typedef TTHStorage THStorage;
        typedef TStorageBase StorageBase;
    };

    template <class TTHTensor, class TStorage, class TSizeStorage>
    class TensorTrait
    {
    public:
        typedef TTHTensor THTensor;
        typedef TStorage Storage;
        typedef TSizeStorage SizeStorage;
    };
}

typedef nn::StorageTrait<THLongStorage, long> StorageLong;
typedef nn::StorageTrait<THFloatStorage, float> StorageFloat;
typedef nn::TensorTrait<THLongTensor, StorageLong, StorageLong> TensorLong;
typedef nn::TensorTrait<THFloatTensor, StorageFloat, StorageLong> TensorFloat;
