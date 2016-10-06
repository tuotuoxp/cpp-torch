#pragma once
#include "Container.h"

#include <vector>


namespace nn
{
    template<class TTensor>
    class Concat : public Container<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Concat"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;

    protected:
        int dimension_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Concat<TTensor>::forward(const nn::Tensor<TTensor> &input)
{
    bool first = true;
    std::vector<long> outputSize;
    std::vector<nn::Tensor<TTensor>> outs;
    for (auto &it : this->modules_)
    {
        nn::Tensor<TTensor> currentOutput = it->forward(input);
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
    nn::Tensor<TTensor> output(true);
    output.resize(outputSize);

    int offset = 0;
    for (size_t i = 0; i < outs.size(); i++)
    {
        nn::Tensor<TTensor> &currentOutput = outs[i];
        output.narrow(dimension_, offset, currentOutput.size(dimension_)).copy(currentOutput);
        offset += currentOutput.size(dimension_);
    }
    return output;
}
