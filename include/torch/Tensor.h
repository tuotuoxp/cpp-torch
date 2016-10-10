#pragma once
#include "../General.h"
#include "Storage.h"

#include <vector>


namespace cpptorch
{
    template<class TTensor>
    class API Tensor
    {
    public:
        explicit Tensor(bool auto_create = false);
        Tensor(const Tensor<TTensor> &other) : th_(nullptr) { *this = other; }
        Tensor(Tensor<TTensor> &&other) : th_(nullptr) { *this = std::move(other); }
        ~Tensor();

        Tensor<TTensor>& operator = (const Tensor<TTensor> &src);
        Tensor<TTensor>& operator = (Tensor<TTensor> &&src);
        operator typename TTensor::TH* () const { return th_; }

        const std::string name() const;

        // creation methods
        void create();
        void create(const Storage<typename TTensor::Storage> &storage, long storage_offset,
            const Storage<typename TTensor::SizeStorage> &size);
        void create(const Storage<typename TTensor::Storage> &storage, long storage_offset,
            const Storage<typename TTensor::SizeStorage> &size, const Storage<typename TTensor::SizeStorage> &stride);
        void resize(const Storage<typename TTensor::SizeStorage> &size);
        void resize(const Storage<typename TTensor::SizeStorage> &size, const Storage<typename TTensor::SizeStorage> &stride);
        void resizeAs(const Tensor<TTensor> &src);
        void copy(const Tensor<TTensor> &src);

        // direct access methods
        bool valid() const { return th_ != nullptr; }
        Storage<typename TTensor::Storage> storage() const;
        long storageOffset() const;
        std::vector<long> size() const;
        long size(int dim) const;
        std::vector<long> stride() const;
        int dim() const;
        typename TTensor::Storage::Base* data() const;
        // for tensor with only ONE element, return it
        operator typename TTensor::Storage::Base() const;

        // calculative access methods
        bool isContiguous() const;
        int nElement() const;

        // special access methods
        // tensor returned by following methods share the same storage
        Tensor<TTensor> narrow(int dimension, long firstIndex, long size) const;
        Tensor<TTensor> select(int dimension, long sliceIndex) const;
        template<class TIterator>
        Tensor<TTensor> view(const TIterator begin, const TIterator end) const;
        template<class TContainer>
        Tensor<TTensor> view(const TContainer &c) const { return view(c.begin(), c.end()); }
        Tensor<TTensor> view(const std::initializer_list<long> &i) const { return view(i.begin(), i.end()); }
        Tensor<TTensor> expand(const std::vector<long> &size) const;
        template<class TIterator>
        Tensor<TTensor> expand(const TIterator begin, const TIterator end) const { return expand(std::vector<long>(begin, end)); }
        template<class TContainer>
        Tensor<TTensor> expand(const TContainer &c) const { return expand(std::vector<long>(c.begin(), c.end())); }
        Tensor<TTensor> expand(const std::initializer_list<long> &i) const { return expand(i.begin(), i.end()); }
        Tensor<TTensor> t() const;
        Tensor<TTensor> operator [] (const std::initializer_list<long> &inputs) const;
        Tensor<TTensor> operator [] (long dim) const { return (*this)[{ dim }]; }

        // math ops (modify tensor itself)
        void fill(typename TTensor::Storage::Base val);
        void abs();
        void addmv(typename TTensor::Storage::Base beta, const Tensor<TTensor> &t,
            typename TTensor::Storage::Base alpha, const Tensor<TTensor> &matrix, const Tensor<TTensor> &vector);
        void addmv(typename TTensor::Storage::Base alpha, const Tensor<TTensor> &matrix, const Tensor<TTensor> &vector)
        {
            addmv(1, *this, alpha, matrix, vector);
        }
        void addmm(typename TTensor::Storage::Base beta, const Tensor<TTensor> &t,
            typename TTensor::Storage::Base alpha, const Tensor<TTensor> &matrix1, const Tensor<TTensor> &matrix2);
        void addr(typename TTensor::Storage::Base beta, const Tensor<TTensor> &t,
            typename TTensor::Storage::Base alpha, const Tensor<TTensor> &vector1, const Tensor<TTensor> &vector2);
        void addr(typename TTensor::Storage::Base alpha, const Tensor<TTensor> &vector1, const Tensor<TTensor> &vector2)
        {
            addr(1, *this, alpha, vector1, vector2);
        }
        Tensor<TTensor>& operator += (typename TTensor::Storage::Base val);
        Tensor<TTensor>& operator += (const Tensor<TTensor> &other);
        Tensor<TTensor>& operator -= (const Tensor<TTensor> &other);
        Tensor<TTensor>& operator *= (typename TTensor::Storage::Base val);
        Tensor<TTensor>& operator *= (const Tensor<TTensor> &other);
        Tensor<TTensor>& operator ^= (typename TTensor::Storage::Base val);
        Tensor<TTensor>& operator ^= (const Tensor<TTensor> &other);

        // tensor math ops (donot modify tensor itself)
        typename TTensor::Storage::Base minall() const;
        typename TTensor::Storage::Base maxall() const;
        Tensor<TTensor> max(int dimension) const;
        Tensor<TTensor> sum(int dimension) const;
        Tensor<TTensor> operator + (typename TTensor::Storage::Base val) const;
        Tensor<TTensor> operator + (const Tensor<TTensor> &other) const;
        Tensor<TTensor> operator - (typename TTensor::Storage::Base val) const;
        Tensor<TTensor> operator - (const Tensor<TTensor> &other) const;
        Tensor<TTensor> operator / (const Tensor<TTensor> &other) const;
        Tensor<TTensor> operator ^ (typename TTensor::Storage::Base val) const;

    protected:
        typename TTensor::TH *th_;
    };


    template<class TTensor>
    API cpptorch::Tensor<TTensor> abs(const cpptorch::Tensor<TTensor> &t);
}
