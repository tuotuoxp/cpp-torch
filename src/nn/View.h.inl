#pragma once
#include "../../include/nn/View.h"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::View<T>::forward(const cpptorch::Tensor<T> &input) const
{
    int ind = input.dim();
    std::vector<long> isz = input.size();
    int maxdim = num_input_dims_ > 0 ? num_input_dims_ : ind;
    int ine = 1;
    for (int i = ind - 1; i >= ind - maxdim; i--)
    {
        ine *= isz[i];
    }

    assert(ine % num_elements_ == 0 && "input view (input) and desired view (size_) do not match");

    // the remainder is either the batch...
    int bsz = ine / num_elements_;
    // ... or the missing size dim
    for (int i = 0; i < (int)size_.size(); i++)
    {
        if (size_[i] == -1)
        {
            bsz = 1;
            break;
        }
    }

    // for dim over maxdim, it is definitively the batch
    for (int i = ind - maxdim - 1; i >= 0; i--)
    {
        bsz *= isz[i];
    }

    // special card
    cpptorch::Tensor<T> output;
    if (bsz == 1 && (num_elements_ <= 0 || ind <= num_elements_))
    {
        output = input.view(size_);
    }
    else
    {
        std::vector<long> sz;
        sz.push_back(bsz);
        sz.insert(sz.end(), size_.begin(), size_.end());
        output = input.view(sz);
    }
    return output;
}
