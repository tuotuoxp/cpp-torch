#pragma once
#include "../General.h"
#include "Storage.h"

#include <vector>


namespace cpptorch
{
    template<typename T>
    class API Tensor
    {
    public:
        explicit Tensor(bool auto_create = false);
        Tensor(const Tensor<T> &other) : th_(nullptr) { *this = other; }
        Tensor(Tensor<T> &&other) : th_(nullptr) { *this = std::move(other); }
        ~Tensor();

        Tensor<T>& operator = (const Tensor<T> &src);
        Tensor<T>& operator = (Tensor<T> &&src);
        operator typename THTrait<T>::Tensor* () const { return th_; }

        const std::string name() const;

        // creation methods
        void create();
        void create(const Storage<T> &storage, long storage_offset, const Storage<long> &size);
        void create(const Storage<T> &storage, long storage_offset, const Storage<long> &size, const Storage<long> &stride);
        void resize(const Storage<long> &size);
        void resize(const Storage<long> &size, const Storage<long> &stride);
        void resizeAs(const Tensor<T> &src);
        void copy(const Tensor<T> &src);

        // direct access methods
        bool valid() const { return th_ != nullptr; }
        Storage<T> storage() const;
        long storageOffset() const;
        std::vector<long> size() const;
        long size(int dim) const;
        std::vector<long> stride() const;
        int dim() const;
        T* data() const;
        // for tensor with only ONE element, return it
        operator T() const;

        // calculative access methods
        bool isContiguous() const;
        int nElement() const;

        // special access methods
        // tensor returned by following methods share the same storage
        Tensor<T> narrow(int dimension, long firstIndex, long size) const;
        Tensor<T> select(int dimension, long sliceIndex) const;
        template<class TIterator>
        Tensor<T> view(const TIterator begin, const TIterator end) const;
        template<class TContainer>
        Tensor<T> view(const TContainer &c) const { return view(c.begin(), c.end()); }
        Tensor<T> view(const std::initializer_list<long> &i) const { return view(i.begin(), i.end()); }
        Tensor<T> expand(const std::vector<long> &size) const;
        template<class TIterator>
        Tensor<T> expand(const TIterator begin, const TIterator end) const { return expand(std::vector<long>(begin, end)); }
        template<class TContainer>
        Tensor<T> expand(const TContainer &c) const { return expand(std::vector<long>(c.begin(), c.end())); }
        Tensor<T> expand(const std::initializer_list<long> &i) const { return expand(i.begin(), i.end()); }
        Tensor<T> t() const;
        Tensor<T> operator [] (const std::initializer_list<long> &inputs) const;
        Tensor<T> operator [] (long dim) const { return (*this)[{ dim }]; }

        // math ops (modify tensor itself)
        void fill(T val);
        void abs();
        void addmv(T beta, const Tensor<T> &t,
            T alpha, const Tensor<T> &matrix, const Tensor<T> &vector);
        void addmv(T alpha, const Tensor<T> &matrix, const Tensor<T> &vector)
        {
            addmv(1, *this, alpha, matrix, vector);
        }
        void addmm(T beta, const Tensor<T> &t,
            T alpha, const Tensor<T> &matrix1, const Tensor<T> &matrix2);
        void addr(T beta, const Tensor<T> &t,
            T alpha, const Tensor<T> &vector1, const Tensor<T> &vector2);
        void addr(T alpha, const Tensor<T> &vector1, const Tensor<T> &vector2)
        {
            addr(1, *this, alpha, vector1, vector2);
        }
        Tensor<T>& operator += (T val);
        Tensor<T>& operator += (const Tensor<T> &other);
        Tensor<T>& operator -= (const Tensor<T> &other);
        Tensor<T>& operator *= (T val);
        Tensor<T>& operator *= (const Tensor<T> &other);
        Tensor<T>& operator ^= (T val);
        Tensor<T>& operator ^= (const Tensor<T> &other);

        // tensor math ops (donot modify tensor itself)
        T minall() const;
        T maxall() const;
        Tensor<T> max(int dimension) const;
        Tensor<T> sum(int dimension) const;
        Tensor<T> operator + (T val) const;
        Tensor<T> operator + (const Tensor<T> &other) const;
        Tensor<T> operator - (T val) const;
        Tensor<T> operator - (const Tensor<T> &other) const;
        Tensor<T> operator / (const Tensor<T> &other) const;
        Tensor<T> operator ^ (T val) const;

    protected:
        typename THTrait<T>::Tensor *th_;
    };


    template<typename T>
    API cpptorch::Tensor<T> abs(const cpptorch::Tensor<T> &t);
}

template<typename T>
API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<T> &m);
