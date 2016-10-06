#pragma once
#include "Layer.h"

#include <cmath>


namespace nn
{
    template<class TTensor>
    class Normalize : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Normalize"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;

    protected:
        typename TTensor::Storage::StorageBase p_, eps_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Normalize<TTensor>::forward(const nn::Tensor<TTensor> &input)
{
    assert(input.dim() <= 2 && "only 1d layer supported");
    std::vector<long> input_size = input.size();
    nn::Tensor<TTensor> input_new;
    if (input.dim() == 1)
    {
        input_new = input.view({ 1, -1 });
    }
    else
    {
        input_new = input;
    }

    nn::Tensor<TTensor> norm;
    if (std::isinf(p_))
    {
        // specialization for the infinity norm
        norm = nn::abs(input_new).max(1) + eps_;
    }
    else
    {
        nn::Tensor<TTensor> buffer;
        if ((int)p_ % 2 != 0)
        {
            buffer = nn::abs(input_new) ^ p_;
        }
        else
        {
            buffer = input_new ^ p_;
        }
        norm = (buffer.sum(1) + eps_) ^ (1 / p_);
    }

    nn::Tensor<TTensor> output;
    output = (input / norm.view({ -1, 1 }).expand(input_new.size())).view(input_size);
    return output;
}
