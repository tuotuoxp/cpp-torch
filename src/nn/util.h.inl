#pragma once
#include "../../include/torch/Tensor.h"
#include "../util.h"


namespace cpptorch
{
    /**
     * add a dimension to an existing tensor
     *
     * before: axbxcxd
     *  after: dim = 0 : 1xaxbxcxd
     *         dim = 2 : axbx1xcxd
     */
    template<typename T>
    cpptorch::Tensor<T> addSingletonDimension(const cpptorch::Tensor<T> &t, int dim)
    {
        int tdim = t.dim();
        asserter(dim > 0 && dim <= tdim + 1)
            << "invalid dimension: " << dim << ". Tensor is of " << tdim << " dimensions.";

        std::vector<long> tsize = t.size(), tstride = t.stride();
        std::vector<long> size, stride;
        size.resize(tdim + 1);
        stride.resize(tdim + 1);

        for (int d = 0; d < dim - 1; d++)
        {
            size[d] = tsize[d];
            stride[d] = tstride[d];
        }
        size[dim] = 1;
        stride[dim] = 1;
        for (int d = dim + 1; d < tdim + 1; d++)
        {
            size[d] = tsize[d - 1];
            stride[d] = tstride[d - 1];
        }
        cpptorch::Tensor<T> view;
        view.create(t.storage(), t.storageOffset(), size, stride);
        return view;
    }

    /**
    * add a dimension to the first of an existing tensor
    *
    * before: axbxc
    *  after: 1xaxbxc
    */
    template<typename T>
    cpptorch::Tensor<T> toBatch(const cpptorch::Tensor<T> &tensor)
    {
        std::vector<long> tsize = tensor.size();
        tsize.insert(tsize.begin(), 1);
        return tensor.view(tsize);
    }

    /**
    * remove the first dimension from an existing tensor
    *
    * before: 1xaxbxc
    *  after: axbxc
    */
    template<typename T>
    cpptorch::Tensor<T> fromBatch(const cpptorch::Tensor<T> &tensor)
    {
        std::vector<long> tsize = tensor.size();
        assert(tsize[0] == 1);
        tsize.erase(tsize.begin());
        return tensor.view(tsize);
    }
}
