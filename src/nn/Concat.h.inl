#pragma once
#include "../../include/nn/Concat.h"


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> cpptorch::nn::Concat<T, F>::forward(const cpptorch::Tensor<T, F> &input) const
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
            outputSize[dimension_] += currentOutput.size(dimension_);
        }
    }
    cpptorch::Tensor<T, F> output(true);
    output.resize(outputSize);

    int offset = 0;
    for (size_t i = 0; i < outs.size(); i++)
    {
        cpptorch::Tensor<T, F> &currentOutput = outs[i];
        output.narrow(dimension_, offset, currentOutput.size(dimension_)).copy(currentOutput);
        offset += currentOutput.size(dimension_);
    }
    return output;
}
