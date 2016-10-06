#pragma once
#include "General.h"

#include <memory>


namespace nn
{
    template<class TStorage>
    class Storage
    {
    public:
        explicit Storage(typename TStorage::THStorage *th = nullptr);
        template<class TIterator>
        Storage(TIterator begin, TIterator end) : th_(nullptr) { unserialze(begin, end); }
        template<class TContainer>
        Storage(const TContainer &c) : th_(nullptr) { unserialze(c); }
        Storage(const std::initializer_list<typename TStorage::StorageBase> &inputs) : th_(nullptr) { unserialze(inputs); }
        Storage(const Storage<TStorage> &other) : th_(nullptr) { *this = other; }
        Storage(Storage<TStorage> &&other) : th_(nullptr) { *this = std::move(other); }
        ~Storage();

        Storage<TStorage>& operator = (const Storage<TStorage> &src);
        Storage<TStorage>& operator = (Storage<TStorage> &&src);
        operator typename TStorage::THStorage* () const { return th_; }

        // getter
        bool valid() const { return th_ != nullptr; }
        int size() const;
        const typename TStorage::StorageBase* data() const;
        typename TStorage::StorageBase* data();

        // from raw ptr
        void unserialze(const typename TStorage::StorageBase *ptr_src, long size, bool take_ownership_of_data = true);
        // from stl iterator
        template<class TIterator>
        void unserialze(const TIterator begin, const TIterator end);
        // from stl container
        template<class TContainer>
        void unserialze(const TContainer &c) { return unserialze(c.begin(), c.end()); }
        // from initializer list
        void unserialze(const std::initializer_list<typename TStorage::StorageBase> &i) { return unserialze(i.begin(), i.end()); }

    protected:
        static void *mallocWrapper(void* ctx, long size) { return malloc(size); }
        static void *reallocWrapper(void* ctx, void* ptr, long size) { return realloc(ptr, size); }
        static void freeWrapper(void* ctx, void* ptr) { free(ptr); }
        static THAllocator allocater_;

        typename TStorage::THStorage *th_;
   };
}
