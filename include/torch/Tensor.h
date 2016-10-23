#pragma once
#include "../General.h"
#include "Storage.h"

#include <vector>


namespace cpptorch
{
    template<typename T, bool C = false>
    class API Tensor
    {
    public:
        explicit Tensor(bool auto_create = false);
        Tensor(const Tensor<T,C> &other) : th_(nullptr) { *this = other; }
        Tensor(Tensor<T,C> &&other) : th_(nullptr) { *this = std::move(other); }
        ~Tensor();

        Tensor<T,C>& operator = (const Tensor<T,C> &src);
        Tensor<T,C>& operator = (Tensor<T,C> &&src);
        operator typename THTrait<T,C>::Tensor* () const { return th_; }

        const std::string name() const;

        // creation methods
        void create();
        void create(const Storage<T,C> &storage, long storage_offset, int dim, const long *size, const long *stride = nullptr);
        void resize(const Storage<long, false> &size);
        void resize(const Storage<long, false> &size, const Storage<long, false> &stride);
        void resizeAs(const Tensor<T,C> &src);
        void copy(const Tensor<T,C> &src);

        // direct access methods
        bool valid() const { return th_ != nullptr; }
        Storage<T,C> storage() const;
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
        Tensor<T,C> narrow(int dimension, long firstIndex, long size) const;
        Tensor<T,C> select(int dimension, long sliceIndex) const;
        template<class TIterator>
        Tensor<T,C> view(const TIterator begin, const TIterator end) const;
        template<class TContainer>
        Tensor<T,C> view(const TContainer &c) const { return view(c.begin(), c.end()); }
        Tensor<T,C> view(const std::initializer_list<long> &i) const { return view(i.begin(), i.end()); }
        Tensor<T,C> expand(const std::vector<long> &size) const;
        template<class TIterator>
        Tensor<T,C> expand(const TIterator begin, const TIterator end) const { return expand(std::vector<long>(begin, end)); }
        template<class TContainer>
        Tensor<T,C> expand(const TContainer &c) const { return expand(std::vector<long>(c.begin(), c.end())); }
        Tensor<T,C> expand(const std::initializer_list<long> &i) const { return expand(i.begin(), i.end()); }
        Tensor<T,C> t() const;
        Tensor<T,C> operator [] (const std::initializer_list<long> &inputs) const;
        Tensor<T,C> operator [] (long dim) const { return (*this)[{ dim }]; }

        // math ops (modify tensor itself)
        void fill(T val);
        void abs();
        void addmv(T beta, const Tensor<T,C> &t,
            T alpha, const Tensor<T,C> &matrix, const Tensor<T,C> &vector);
        void addmv(T alpha, const Tensor<T,C> &matrix, const Tensor<T,C> &vector)
        {
            addmv(1, *this, alpha, matrix, vector);
        }
        void addmm(T beta, const Tensor<T,C> &t,
            T alpha, const Tensor<T,C> &matrix1, const Tensor<T,C> &matrix2);
        void addr(T beta, const Tensor<T,C> &t,
            T alpha, const Tensor<T,C> &vector1, const Tensor<T,C> &vector2);
        void addr(T alpha, const Tensor<T,C> &vector1, const Tensor<T,C> &vector2)
        {
            addr(1, *this, alpha, vector1, vector2);
        }
        Tensor<T,C>& operator += (T val);
        Tensor<T,C>& operator += (const Tensor<T,C> &other);
        Tensor<T,C>& operator -= (const Tensor<T,C> &other);
        Tensor<T,C>& operator *= (T val);
        Tensor<T,C>& operator *= (const Tensor<T,C> &other);
        Tensor<T,C>& operator ^= (T val);
        Tensor<T,C>& operator ^= (const Tensor<T,C> &other);

        // tensor math ops (donot modify tensor itself)
        T minall() const;
        T maxall() const;
        Tensor<T,C> max(int dimension) const;
        Tensor<T,C> sum(int dimension) const;
        Tensor<T,C> operator + (T val) const;
        Tensor<T,C> operator + (const Tensor<T,C> &other) const;
        Tensor<T,C> operator - (T val) const;
        Tensor<T,C> operator - (const Tensor<T,C> &other) const;
        Tensor<T,C> operator / (const Tensor<T,C> &other) const;
        Tensor<T,C> operator ^ (T val) const;

    protected:
        typename THTrait<T,C>::Tensor *th_;
    };


    template<typename T, bool C>
    API cpptorch::Tensor<T,C> abs(const cpptorch::Tensor<T,C> &t);
}

template<typename T, bool C>
API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<T,C> &m);
