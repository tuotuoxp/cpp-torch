#pragma once
#include "../General.h"

#include <stdlib.h>
#include <memory>


namespace cpptorch
{
    template<typename T>
    class Storage
    {
    public:
        explicit Storage(typename THTrait<T>::Storage *th = nullptr);
        template<class TIterator>
        Storage(TIterator begin, TIterator end) : th_(nullptr) { unserialze(begin, end); }
        template<class TContainer>
        Storage(const TContainer &c) : th_(nullptr) { unserialze(c); }
        Storage(const std::initializer_list<T> &inputs) : th_(nullptr) { unserialze(inputs); }
        Storage(const Storage<T> &other) : th_(nullptr) { *this = other; }
        Storage(Storage<T> &&other) : th_(nullptr) { *this = std::move(other); }
        ~Storage();

        Storage<T>& operator = (const Storage<T> &src);
        Storage<T>& operator = (Storage<T> &&src);
        operator typename THTrait<T>::Storage* () const { return th_; }

        // getter
        bool valid() const { return th_ != nullptr; }
        int size() const;
        const T* data() const;
        T* data();

        // creator
        void create();
        // from raw ptr
        void unserialze(const T *ptr_src, long size, bool take_ownership_of_data = true);
        // from stl iterator
        template<class TIterator>
        void unserialze(const TIterator begin, const TIterator end)
        {
            long size = (long)(end - begin);
            T *ptr = (T*)malloc(size * sizeof(T));
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
        void unserialze(const std::initializer_list<T> &i) { return unserialze(i.begin(), i.end()); }

    protected:
        typename THTrait<T>::Storage *th_;
   };
}
