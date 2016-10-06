#pragma once
#include "Layer.h"


namespace nn
{
    template<class TTensor>
    class MulConstant : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.MulConstant"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) override;

    protected:
        bool inplace_;
        typename TTensor::Storage::StorageBase constant_scalar_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::MulConstant<TTensor>::forward(const nn::Tensor<TTensor> &input)
{
    nn::Tensor<TTensor> output(true);
    output.resizeAs(input);
    output.copy(input);
    output *= constant_scalar_;
    return output;
}
