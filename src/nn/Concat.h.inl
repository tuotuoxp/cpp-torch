#pragma once
#include "../../include/nn/Concat.h"


template<class TTensor>
cpptorch::Tensor<TTensor> cpptorch::nn::Concat<TTensor>::forward(const cpptorch::Tensor<TTensor> &input) const
{
    bool first = true;
    std::vector<long> outputSize;
    std::vector<cpptorch::Tensor<TTensor>> outs;
    for (auto &it : this->modules_)
    {
        cpptorch::Tensor<TTensor> currentOutput = it->forward(input);
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
    cpptorch::Tensor<TTensor> output(true);
    output.resize(outputSize);

    int offset = 0;
    for (size_t i = 0; i < outs.size(); i++)
    {
        cpptorch::Tensor<TTensor> &currentOutput = outs[i];
        output.narrow(dimension_, offset, currentOutput.size(dimension_)).copy(currentOutput);
        offset += currentOutput.size(dimension_);
    }
    return output;
}
