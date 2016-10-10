#include "../include/cpptorch.h"
#include "torch/Storage.h.inl"
#include "torch/Tensor.h.inl"
#include "torch/TensorPrint.h.inl"


template API class cpptorch::Storage<StorageLong>;
template API class cpptorch::Storage<StorageFloat>;
template API class cpptorch::Storage<StorageDouble>;
template API class cpptorch::Tensor<TensorLong>;
template API class cpptorch::Tensor<TensorFloat>;
template API class cpptorch::Tensor<TensorDouble>;

template API cpptorch::Tensor<TensorLong> cpptorch::abs(const cpptorch::Tensor<TensorLong> &t);
template API cpptorch::Tensor<TensorFloat> cpptorch::abs(const cpptorch::Tensor<TensorFloat> &t);
template API cpptorch::Tensor<TensorDouble> cpptorch::abs(const cpptorch::Tensor<TensorDouble> &t);

template API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<TensorLong> &m);
template API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<TensorFloat> &m);
template API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<TensorDouble> &m);


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


template API cpptorch::Tensor<TensorFloat> cpptorch::read_tensor(const object *obj);
template API cpptorch::Tensor<TensorDouble> cpptorch::read_tensor(const object *obj);
template API std::shared_ptr<cpptorch::nn::Layer<TensorFloat>> cpptorch::read_net(const object *obj);
template API std::shared_ptr<cpptorch::nn::Layer<TensorDouble>> cpptorch::read_net(const object *obj);
