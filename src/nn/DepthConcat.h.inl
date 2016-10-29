#pragma once
#include "../../include/nn/DepthConcat.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::DepthConcat<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
{
    bool first = true;
    std::vector<long> outputSize;
    std::vector<cpptorch::Tensor<T, F>> outs;
    for (auto &it : this->modules_)
    {
        cpptorch::Tensor<T, F> currentOutput = it->forward(input);
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

    cpptorch::Tensor<T, F> output(true);
    output.resize(outputSize);
    output.fill(0);

    int offset = 0;
    for (size_t i = 0; i < outs.size(); i++)
    {
        cpptorch::Tensor<T, F> &currentOutput = outs[i];
        cpptorch::Tensor<T, F> outputWindow = windowNarrow(output, currentOutput, outputSize, offset);
        outputWindow.copy(currentOutput);
        offset += currentOutput.size(this->dimension_);
    }
    return output;
}


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::DepthConcat<T, F>::windowNarrow(cpptorch::Tensor<T, F> &output, cpptorch::Tensor<T, F> &currentOutput,
    std::vector<long> &outputSize, int offset) const
{
    cpptorch::Tensor<T, F> outputWindow = output.narrow(this->dimension_, offset, currentOutput.size(this->dimension_));
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
