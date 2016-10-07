#pragma once
#include "Sequential.h"


namespace nn
{
    template<class TTensor>
    class Threshold : public Layer<TTensor>
    {
    public:
        virtual const std::string name() const override { return "nn.Threshold"; }
        virtual Tensor<TTensor> forward(const Tensor<TTensor> &input) const override;

    protected:
        typename TTensor::Storage::StorageBase threshold_, val_;
        bool inplace_;
    };
}


template<class TTensor>
nn::Tensor<TTensor> nn::Threshold<TTensor>::forward(const nn::Tensor<TTensor> &input) const
{
    // validate parameters
    if (inplace_)
    {
        asserter(val_ <= threshold_) << "in-place processing requires value (" << val_ << ") not exceed threshold (" << threshold_ << ")";
    }

    nn::Tensor<TTensor> output(true);
    THWrapper::NN<TTensor>::Threshold_updateOutput(nullptr, input, output, threshold_, val_, inplace_);
    return output;
}
