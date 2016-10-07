#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class Reshape : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Reshape"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

    protected:
        int nelement_;
        bool batch_mode_;
        std::vector<long> size_, batchsize_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Reshape<TTensor>::forward(const nn::Tensor<TTensor> &input) const
{
    nn::Tensor<TTensor> input_new;
    if (!input.isContiguous())
    {
        input_new.create();
        input_new.resizeAs(input);
        input_new.copy(input);
    }
    else
    {
        input_new = input;
    }
    nn::Tensor<TTensor> output;
    if (!batch_mode_ && input_new.nElement() == nelement_ && input_new.size(0) != 1)
    {
        output = input_new.view(size_);
    }
    else
    {
        std::vector<long> batchsize = batchsize_;
        batchsize[0] = input_new.size(0);
        output = input_new.view(batchsize);
    }
    return output;
}
