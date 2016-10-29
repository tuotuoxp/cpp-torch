#pragma once
#include "../General.h"
#include "Storage.h"

#include <vector>


namespace cpptorch
{
    template<typename T, GPUFlag F = GPU_None>
    class API Tensor
    {
    public:
        explicit Tensor(bool auto_create = false);
        Tensor(const Tensor<T, F> &other) : th_(nullptr) { *this = other; }
        Tensor(Tensor<T, F> &&other) : th_(nullptr) { *this = std::move(other); }
        ~Tensor();

        Tensor<T, F>& operator = (const Tensor<T, F> &src);
        Tensor<T, F>& operator = (Tensor<T, F> &&src);
        operator typename THTrait<T, F>::Tensor* () const { return th_; }

        const std::string name() const;

        // creation methods
        void create();
        void create(const Storage<T, F> &storage, long storage_offset, int dim, const long *size, const long *stride = nullptr);
        void resize(const std::vector<long> &size);
        void resize(const std::vector<long> &size, const std::vector<long> &stride);
        void resizeAs(const Tensor<T, F> &src);
        void copy(const Tensor<T, F> &src);

        // direct access methods
        bool valid() const { return th_ != nullptr; }
        Storage<T, F> storage() const;
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
        Tensor<T, F> narrow(int dimension, long firstIndex, long size) const;
        Tensor<T, F> select(int dimension, long sliceIndex) const;
        template<class TIterator>
        Tensor<T, F> view(const TIterator begin, const TIterator end) const;
        template<class TContainer>
        Tensor<T, F> view(const TContainer &c) const { return view(c.begin(), c.end()); }
        Tensor<T, F> view(const std::initializer_list<long> &i) const { return view(i.begin(), i.end()); }
        Tensor<T, F> expand(const std::vector<long> &size) const;
        template<class TIterator>
        Tensor<T, F> expand(const TIterator begin, const TIterator end) const { return expand(std::vector<long>(begin, end)); }
        template<class TContainer>
        Tensor<T, F> expand(const TContainer &c) const { return expand(std::vector<long>(c.begin(), c.end())); }
        Tensor<T, F> expand(const std::initializer_list<long> &i) const { return expand(i.begin(), i.end()); }
        Tensor<T, F> t() const;
        Tensor<T, F> operator [] (const std::initializer_list<long> &inputs) const;
        Tensor<T, F> operator [] (long dim) const { return (*this)[{ dim }]; }

        // math ops (modify tensor itself)
        void fill(T val);
        void abs();
        void addmv(T beta, const Tensor<T, F> &t,
            T alpha, const Tensor<T, F> &matrix, const Tensor<T, F> &vector);
        void addmv(T alpha, const Tensor<T, F> &matrix, const Tensor<T, F> &vector)
        {
            addmv(1, *this, alpha, matrix, vector);
        }
        void addmm(T beta, const Tensor<T, F> &t,
            T alpha, const Tensor<T, F> &matrix1, const Tensor<T, F> &matrix2);
        void addr(T beta, const Tensor<T, F> &t,
            T alpha, const Tensor<T, F> &vector1, const Tensor<T, F> &vector2);
        void addr(T alpha, const Tensor<T, F> &vector1, const Tensor<T, F> &vector2)
        {
            addr(1, *this, alpha, vector1, vector2);
        }
        Tensor<T, F>& operator += (T val);
        Tensor<T, F>& operator += (const Tensor<T, F> &other);
        Tensor<T, F>& operator -= (const Tensor<T, F> &other);
        Tensor<T, F>& operator *= (T val);
        Tensor<T, F>& operator *= (const Tensor<T, F> &other);
        Tensor<T, F>& operator ^= (T val);
        Tensor<T, F>& operator ^= (const Tensor<T, F> &other);

        // tensor math ops (donot modify tensor itself)
        T minall() const;
        T maxall() const;
        Tensor<T, F> max(int dimension) const;
        Tensor<T, F> sum(int dimension) const;
        Tensor<T, F> operator + (T val) const;
        Tensor<T, F> operator + (const Tensor<T, F> &other) const;
        Tensor<T, F> operator - (T val) const;
        Tensor<T, F> operator - (const Tensor<T, F> &other) const;
        Tensor<T, F> operator / (const Tensor<T, F> &other) const;
        Tensor<T, F> operator ^ (T val) const;

    protected:
        typename THTrait<T, F>::Tensor *th_;
    };


    template<typename T, GPUFlag F>
    API cpptorch::Tensor<T, F> abs(const cpptorch::Tensor<T, F> &t);
}

template<typename T, GPUFlag F>
API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<T, F> &m);
