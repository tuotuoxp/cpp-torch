#pragma once
#include "../../include/nn/DepthConcat.h"


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::DepthConcat<T,C>::forward(const cpptorch::Tensor<T,C> &input) const
{
    bool first = true;
    std::vector<long> outputSize;
    std::vector<cpptorch::Tensor<T,C>> outs;
    for (auto &it : this->modules_)
    {
        cpptorch::Tensor<T,C> currentOutput = it->forward(input);
        outs.push_back(currentOutput);
        if (first)
        {
            outputSize = currentOutput.size();
            first = false;
        }
        else
        {
            outputSize[this->dimension_] += currentOutput.size(this->dimension_);
            for (int dim = 0; dim < (int)outputSize.size(); dim++)
            {
                if (dim != this->dimension_)
                {
                    // take the maximum size(shouldn't change anything for batch dim)
                    outputSize[dim] = std::max(outputSize[dim], currentOutput.size(dim));
                }
            }
        }
    }

    cpptorch::Tensor<T,C> output(true);
    output.resize(outputSize);
    output.fill(0);

    int offset = 0;
    for (size_t i = 0; i < outs.size(); i++)
    {
        cpptorch::Tensor<T,C> &currentOutput = outs[i];
        cpptorch::Tensor<T,C> outputWindow = windowNarrow(output, currentOutput, outputSize, offset);
        outputWindow.copy(currentOutput);
        offset += currentOutput.size(this->dimension_);
    }
    return output;
}


template<typename T, bool C>
cpptorch::Tensor<T,C> cpptorch::nn::DepthConcat<T,C>::windowNarrow(cpptorch::Tensor<T,C> &output, cpptorch::Tensor<T,C> &currentOutput,
    std::vector<long> &outputSize, int offset) const
{
    cpptorch::Tensor<T,C> outputWindow = output.narrow(this->dimension_, offset, currentOutput.size(this->dimension_));
    for (int dim = 0; dim < (int)outputSize.size(); dim++)
    {
        long currentSize = currentOutput.size(dim);
        if (dim != this->dimension_ && outputSize[dim] != currentSize)
        {
            // 5x5 vs 3x3 -> start = [(5-3)/2] = 1 (1 pad each side)
            // 9x9 vs 5x5 -> start = [(9-5)/2] = 2 (2 pad each side)
            // 9x9 vs 4x4 -> start = [(9-4)/2] = 2.5 (2 pad, 3 pad)
            int start = (int)floor((outputSize[dim] - currentSize) / 2);
            outputWindow = outputWindow.narrow(dim, start, currentSize);
        }
    }
    return outputWindow;
}
