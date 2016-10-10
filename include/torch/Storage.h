#pragma once
#include "../General.h"

#include <memory>


namespace cpptorch
{
    template<class TStorage>
    class Storage
    {
    public:
        explicit Storage(typename TStorage::TH *th = nullptr);
        template<class TIterator>
        Storage(TIterator begin, TIterator end) : th_(nullptr) { unserialze(begin, end); }
        template<class TContainer>
        Storage(const TContainer &c) : th_(nullptr) { unserialze(c); }
        Storage(const std::initializer_list<typename TStorage::Base> &inputs) : th_(nullptr) { unserialze(inputs); }
        Storage(const Storage<TStorage> &other) : th_(nullptr) { *this = other; }
        Storage(Storage<TStorage> &&other) : th_(nullptr) { *this = std::move(other); }
        ~Storage();

        Storage<TStorage>& operator = (const Storage<TStorage> &src);
        Storage<TStorage>& operator = (Storage<TStorage> &&src);
        operator typename TStorage::TH* () const { return th_; }

        // getter
        bool valid() const { return th_ != nullptr; }
        int size() const;
        const typename TStorage::Base* data() const;
        typename TStorage::Base* data();

        // from raw ptr
        void unserialze(const typename TStorage::Base *ptr_src, long size, bool take_ownership_of_data = true);
        // from stl iterator
        template<class TIterator>
        void unserialze(const TIterator begin, const TIterator end)
        {
            long size = (long)(end - begin);
            typename TStorage::Base *ptr = (typename TStorage::Base*)malloc(size * sizeof(typename TStorage::Base));
            int i = 0;
            for (auto it = begin; it != end; it++, i++)
            {
                ptr[i] = *it;
            }
            unserialze(ptr, size);
        }
        // from stl container
        template<class TContainer>
        void unserialze(const TContainer &c) { return unserialze(c.begin(), c.end()); }
        // from initializer list
        void unserialze(const std::initializer_list<typename TStorage::Base> &i) { return unserialze(i.begin(), i.end()); }

    protected:
        typename TStorage::TH *th_;
   };
}
