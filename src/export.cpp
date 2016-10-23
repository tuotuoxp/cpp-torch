#include "../include/cpptorch.h"
#include "torch/Storage.h.inl"
#include "torch/Tensor.h.inl"
#include "torch/TensorPrint.h.inl"


template API class cpptorch::Storage<long, false>;
template API class cpptorch::Storage<float, false>;
template API class cpptorch::Storage<double, false>;
template API class cpptorch::Tensor<long, false>;
template API class cpptorch::Tensor<float, false>;
template API class cpptorch::Tensor<double, false>;

template API cpptorch::Tensor<long, false> cpptorch::abs(const cpptorch::Tensor<long, false> &t);
template API cpptorch::Tensor<float, false> cpptorch::abs(const cpptorch::Tensor<float, false> &t);
template API cpptorch::Tensor<double, false> cpptorch::abs(const cpptorch::Tensor<double, false> &t);

template API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<long, false> &m);
template API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<float, false> &m);
template API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<double, false> &m);


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


#include "builder.h.inl"
#include "reader.h.inl"


template API cpptorch::Tensor<float, false> cpptorch::read_tensor(const object *obj);
template API cpptorch::Tensor<double, false> cpptorch::read_tensor(const object *obj);
template API std::shared_ptr<cpptorch::nn::Layer<float, false>> cpptorch::read_net(const object *obj);
template API std::shared_ptr<cpptorch::nn::Layer<double, false>> cpptorch::read_net(const object *obj);
