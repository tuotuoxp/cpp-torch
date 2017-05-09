#include "../include/cpptorch_cuda.h"
#include "../src/torch/Storage.h.inl"
#include "../src/torch/Tensor.h.inl"
#include "../src/torch/TensorPrint.h.inl"
#include "th_wrapper.h"


template API class cpptorch::Storage<long, GPU_Cuda>;
template API class cpptorch::Storage<float, GPU_Cuda>;
template API class cpptorch::Storage<double, GPU_Cuda>;
template API class cpptorch::Tensor<long, GPU_Cuda>;
template API class cpptorch::Tensor<float, GPU_Cuda>;
template API class cpptorch::Tensor<double, GPU_Cuda>;

template API cpptorch::Tensor<long, GPU_Cuda> cpptorch::abs(const cpptorch::Tensor<long, GPU_Cuda> &t);
template API cpptorch::Tensor<float, GPU_Cuda> cpptorch::abs(const cpptorch::Tensor<float, GPU_Cuda> &t);
template API cpptorch::Tensor<double, GPU_Cuda> cpptorch::abs(const cpptorch::Tensor<double, GPU_Cuda> &t);


template<> API std::ostream& operator << (std::ostream &o, const cpptorch::Tensor<float, GPU_Cuda> &t)
{
    cpptorch::Tensor<float> t_cpu(true);
    cpptorch::th::copy_cuda2cpu<float>(t_cpu, t);
    return TensorPrint<float>(o, t_cpu).printTensor(t.name()) << std::endl;
}


#include "../src/nn/BatchNormalization.h.inl"
#include "../src/nn/Concat.h.inl"
#include "../src/nn/Container.h.inl"
#include "../src/nn/Decorator.h.inl"
#include "../src/nn/DepthConcat.h.inl"
#include "../src/nn/Inception.h.inl"
#include "../src/nn/Linear.h.inl"
#include "../src/nn/Add.h.inl"
#include "../src/nn/MulConstant.h.inl"
#include "../src/nn/Normalize.h.inl"
#include "../src/nn/SoftMax.h.inl"
#include "../src/nn/LogSoftMax.h.inl"
#include "../src/nn/Reshape.h.inl"
#include "../src/nn/Sequential.h.inl"
#include "../src/nn/SpatialAveragePooling.h.inl"
#include "../src/nn/SpatialConvolution.h.inl"
#include "../src/nn/SpatialConvolutionMM.h.inl"
#include "../src/nn/SpatialCrossMapLRN.h.inl"
#include "../src/nn/SpatialMaxPooling.h.inl"
#include "../src/nn/SpatialReflectionPadding.h.inl"
#include "../src/nn/Sqrt.h.inl"
#include "../src/nn/Square.h.inl"
#include "../src/nn/Threshold.h.inl"
#include "../src/nn/View.h.inl"


#include "../src/builder.h.inl"
#include "../src/reader.h.inl"


cpptorch::CudaTensor cpptorch::read_cuda_tensor(const cpptorch::object *obj)
{
    object_reader<float, GPU_Cuda> mb;
    return mb.build_tensor(obj);
}

std::shared_ptr<cpptorch::nn::CudaLayer> cpptorch::read_cuda_net(const cpptorch::object *obj)
{
    object_reader<float, GPU_Cuda> mb;
    return std::static_pointer_cast<cpptorch::nn::Layer<float, GPU_Cuda>>(mb.build_layer(obj));
}


cpptorch::Tensor<float, GPU_Cuda> cpptorch::cpu2cuda(const cpptorch::Tensor<float> &t)
{
    cpptorch::Tensor<float, GPU_Cuda> t_gpu(true);
    cpptorch::th::copy_cpu2cuda<float>(t_gpu, t);
    return t_gpu;
}

cpptorch::Tensor<float> cpptorch::cuda2cpu(const cpptorch::Tensor<float, GPU_Cuda> &t)
{
    cpptorch::Tensor<float> t_cpu(true);
    cpptorch::th::copy_cuda2cpu<float>(t_cpu, t);
    return t_cpu;
}
