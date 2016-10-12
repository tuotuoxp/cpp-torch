#pragma once
#include "../../include/nn/SpatialCrossMapLRN.h"
#include "util.h.inl"


template<typename T>
cpptorch::Tensor<T> cpptorch::nn::SpatialCrossMapLRN<T>::forward(const cpptorch::Tensor<T> &input) const
{
    int idim = input.dim();
    asserter(idim == 3 || idim == 4) << "Input must be 3D or 4D";

    cpptorch::Tensor<T> input_new;
    bool isBatch = true;
    if (input.dim() == 3)
    {
        input_new = cpptorch::addSingletonDimension(input, 1);
        isBatch = false;
    }
    else
    {
        input_new = input;
    }

    std::vector<long> input_size = input_new.size();
    long channels = input_size[1];

    // storage #1 : inputSquare/squareNext/squarePrevious/output
    // storage #2 : scale/scaleFirst/scalePrevious/scaleCurrent
    cpptorch::Tensor<T> inputSquare = input_new ^ (T)2;
    
    int prePad = (size_ - 1) / 2 + 1;
    int prePadCrop = prePad > channels ? channels : prePad;

    cpptorch::Tensor<T> scale(true);
    scale.resizeAs(input_new);

    cpptorch::Tensor<T> scaleFirst = scale.select(1, 0);
    scaleFirst.fill(0);

    // compute first feature map normalization
    for (int c = 0; c < prePadCrop; c++)
    {
        scaleFirst += inputSquare.select(1, c);
    }

    // reuse computations for next feature maps normalization
    // by adding the next feature map and removing the previous
    for (int c = 1; c < channels; c++)
    {
        cpptorch::Tensor<T> scalePrevious = scale.select(1, c - 1);
        cpptorch::Tensor<T> scaleCurrent = scale.select(1, c);
        scaleCurrent.copy(scalePrevious);
        if (c <= channels - prePad)
        {
            cpptorch::Tensor<T> squareNext = inputSquare.select(1, c + prePad - 1);
            scaleCurrent += squareNext;
        }
        if (c >= prePad)
        {
            cpptorch::Tensor<T> squarePrevious = inputSquare.select(1, c - prePad);
            scaleCurrent -= squarePrevious;
        }
    }

    scale *= alpha_ / (T)size_;
    scale += k_;

    // use scale's storage as output buffer
    cpptorch::Tensor<T> output = scale;
    output ^= -beta_;
    output *= input_new;

    if (!isBatch)
    {
        output = output[1];
    }
    return output;

    /*
    void THNN_CudaSpatialCrossMapLRN_updateOutput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *output,
  THCudaTensor *scale,
  int size,
  float alpha,
  float beta,
  float k)
    */
}
