#include "../include/cpptorch.h"
#include "torch/Storage.h.inl"
#include "torch/Tensor.h.inl"
#include "torch/TensorPrint.h.inl"


template API class cpptorch::Storage<long, GPU_None>;
template API class cpptorch::Storage<float, GPU_None>;
template API class cpptorch::Storage<double, GPU_None>;
template API class cpptorch::Tensor<long, GPU_None>;
template API class cpptorch::Tensor<float, GPU_None>;
template API class cpptorch::Tensor<double, GPU_None>;

template API cpptorch::Tensor<long, GPU_None> cpptorch::abs(const cpptorch::Tensor<long, GPU_None> &t);
template API cpptorch::Tensor<float, GPU_None> cpptorch::abs(const cpptorch::Tensor<float, GPU_None> &t);
template API cpptorch::Tensor<double, GPU_None> cpptorch::abs(const cpptorch::Tensor<double, GPU_None> &t);


template<> API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<long, GPU_None> &m)
{
    return TensorPrint<long>(o, m).printTensor(m.name()) << std::endl;
}
template<> API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<float, GPU_None> &m)
{
    return TensorPrint<float>(o, m).printTensor(m.name()) << std::endl;
}
template<> API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<double, GPU_None> &m)
{
    return TensorPrint<double>(o, m).printTensor(m.name()) << std::endl;
}


#include "nn/BatchNormalization.h.inl"
#include "nn/Concat.h.inl"
#include "nn/Container.h.inl"
#include "nn/Decorator.h.inl"
#include "nn/DepthConcat.h.inl"
#include "nn/Inception.h.inl"
#include "nn/Linear.h.inl"
#include "nn/MulConstant.h.inl"
#include "nn/Normalize.h.inl"
#include "nn/Reshape.h.inl"
#include "nn/Sequential.h.inl"
#include "nn/SpatialAveragePooling.h.inl"
#include "nn/SpatialConvolution.h.inl"
#include "nn/SpatialConvolutionMM.h.inl"
#include "nn/SpatialCrossMapLRN.h.inl"
#include "nn/SpatialMaxPooling.h.inl"
#include "nn/SpatialReflectionPadding.h.inl"
#include "nn/Sqrt.h.inl"
#include "nn/Square.h.inl"
#include "nn/Threshold.h.inl"
#include "nn/View.h.inl"


template API class cpptorch::nn::BatchNormalization<float, GPU_None>;
template API class cpptorch::nn::BatchNormalization<double, GPU_None>;


#include "builder.h.inl"
#include "reader.h.inl"


template API class cpptorch::layer_creator<float, GPU_None>;
template API class cpptorch::layer_creator<double, GPU_None>;

template API cpptorch::Tensor<float, GPU_None> cpptorch::read_tensor(const object *obj);
template API cpptorch::Tensor<double, GPU_None> cpptorch::read_tensor(const object *obj);

template API std::shared_ptr<cpptorch::nn::Layer<float, GPU_None>> cpptorch::read_net(const object*, cpptorch::layer_creator<float, GPU_None>*);
template API std::shared_ptr<cpptorch::nn::Layer<double, GPU_None>> cpptorch::read_net(const object*, cpptorch::layer_creator<double, GPU_None>*);
