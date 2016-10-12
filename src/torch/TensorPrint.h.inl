#pragma once
#include "../util.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>


template<typename T>
class TensorPrint
{
public:
    TensorPrint(std::ostream &o, const cpptorch::Tensor<T> &t);
    
    void printTensor();

protected:
    void getPrintFormat(int offset, double &scale, int &len);
    void printMatrix(int offset, const std::string &indent);
    void printSubTensor(int offset, std::vector<int> &dims);
    
    
    std::ostream &out_;
    const cpptorch::Tensor<T> &tensor_;
    const cpptorch::Storage<T> storage_cache_;
    const std::vector<long> stride_cache_;
    const std::vector<long> size_cache_;
    const T * const data_;
    const int offset_;
};


template<typename T>
TensorPrint<T>::TensorPrint(std::ostream &o, const cpptorch::Tensor<T> &t) : out_(o), tensor_(t),
    storage_cache_(tensor_.storage()),
    stride_cache_(tensor_.stride()),
    size_cache_(tensor_.size()),
    data_(storage_cache_.data()),
    offset_(tensor_.storageOffset()) {}


template<typename T>
void TensorPrint<T>::getPrintFormat(int offset, double &scale, int &len)
{
    scale = 1.0f;
    bool is_int = true;
    auto minval = std::numeric_limits<T>::max();
    auto maxval = std::numeric_limits<T>::min();

    int dim_x = (int)size_cache_.size() - 1;
    int size_y = dim_x == 0 ? 1 : size_cache_[dim_x - 1];
    for (int y = 0; y < size_y; y++)
    {
        int ox = offset;
        for (int x = 0; x < size_cache_[dim_x]; x++)
        {
            if (is_int && data_[ox] != ceil(data_[ox]))
            {
                is_int = false;
            }
            minval = std::min(minval, template_abs(data_[ox]));
            maxval = std::max(maxval, template_abs(data_[ox]));
            ox += stride_cache_[dim_x];
        }
        if (dim_x > 0)
        {
            offset += stride_cache_[dim_x - 1];
        }
    }
    
    int exp_min = 1 + (minval != 0 ? (int)floor(log10(minval)) : 0);
    int exp_max = 1 + (maxval != 0 ? (int)floor(log10(maxval)) : 0);
    
    out_.precision(4);
    out_ << std::fixed;
    if (is_int)
    {
        if (exp_max > 9)
        {
            out_ << std::scientific;
            len = 11;
        }
        else
        {
            out_.precision(0);
            len = std::max(exp_max, 3) + 1;
        }
    }
    else
    {
        if (exp_max - exp_min > 4)
        {
            out_ << std::scientific;
            len = 11;
            if (std::abs(exp_max) > 99 || std::abs(exp_min) > 99)
            {
                len = len + 1;
            }
        }
        else
        {
            if (exp_max > 5 || exp_max < 0)
            {
                len = 7;
                scale = pow(10, exp_max - 1);
            }
            else
            {
                if (exp_max == 0)
                {
                    len = 7;
                }
                else
                {
                    len = exp_max + 6;
                }
            }
        }
    }
}


template<typename T>
void TensorPrint<T>::printMatrix(int offset, const std::string &indent)
{
    double scale;
    int len;
    getPrintFormat(offset, scale, len);
    std::ios state(nullptr);
    state.copyfmt(out_);

    int dim_x = (int)size_cache_.size() - 1;
    int column_per_line = (int)floor((80 - indent.length()) / (len + 2));
    bool not_enough_col = column_per_line < size_cache_[dim_x];
    int y_begin = 0;
    std::string data_indent = not_enough_col ? (indent + " ") : indent;
    while (y_begin < size_cache_[dim_x])
    {
        int y_end = std::min(y_begin + column_per_line, (int)size_cache_[dim_x]);
        if (not_enough_col)
        {
            if (y_begin != 0)
            {
                out_ << indent << std::endl;
            }
            out_ << indent << "Columns " << y_begin << " to " << (y_end - 1) << std::endl;
        }
        if (scale != 1)
        {
            out_.precision(8);
            out_.unsetf(std::ios_base::floatfield);
            out_ << data_indent << scale << " *" << std::endl;
            out_.copyfmt(state);
        }
        int offset_round = offset;
        for (int y = 0; y < size_cache_[dim_x - 1]; y++)
        {
            out_ << data_indent;
            int ox = offset_round + y_begin * stride_cache_[dim_x];
            for (int x = y_begin; x < y_end; x++)
            {
                out_ << (data_[ox] >= 0 ? " " : "") << data_[ox] / scale;
                if (x < y_end - 1)
                {
                    out_ << " ";
                }
                else
                {
                    out_ << std::endl;
                }
                ox += stride_cache_[dim_x];
            }
            offset_round += stride_cache_[dim_x - 1];
        }
        y_begin = y_end;
    }
}


template<typename T>
void TensorPrint<T>::printSubTensor(int offset, std::vector<int> &dims)
{
    if (dims.size() == size_cache_.size() - 2)
    {
        if (offset != offset_)
        {
            // not the first one
            out_ << std::endl;
        }
        out_ << "(";
        std::copy(dims.begin(), dims.end(), std::ostream_iterator<long>(out_, ","));
        out_ << ".,.) = " << std::endl;
        printMatrix(offset, "  ");
        return;
    }
    
    int current_dim = (int)dims.size();
    dims.push_back(0);
    for (int i = 0; i < (int)size_cache_[current_dim]; i++)
    {
        dims[current_dim] = i;
        printSubTensor(offset + i * stride_cache_[current_dim], dims);
    }
    dims.pop_back();
}


template<typename T>
void TensorPrint<T>::printTensor()
{
    std::ios state_init(nullptr);
    state_init.copyfmt(out_);
    if (size_cache_.size() == 1)
    {
        double scale;
        int len;
        getPrintFormat(offset_, scale, len);
        std::ios state(nullptr);
        state.copyfmt(out_);

        if (scale != 1)
        {
            out_.precision(8);
            out_.unsetf(std::ios_base::floatfield);
            out_ << scale << " *" << std::endl;
            out_.copyfmt(state);
        }
        int ox = offset_;
        for (int x = 0; x < size_cache_[0]; x++)
        {
            out_ << (data_[ox] >= 0 ? " " : "") << data_[ox] / scale << std::endl;
            ox += stride_cache_[0];
        }
    }
    else
    {
        std::vector<int> dims;
        printSubTensor(offset_, dims);
    }
    out_.copyfmt(state_init);
    // get tensor name
    out_ << "[" << tensor_.name() << " of size " << join(size_cache_, "x") << "]" << std::endl;
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<T> &m)
{
	TensorPrint<T>(o, m).printTensor();
	o << std::endl;
	return o;
}
