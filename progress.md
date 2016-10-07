This list shows current supported modules.

Our purpose is to facilitate the **testing procedure**. So backward functions won't be supported in a foresable future.

# nn package
We follow the structure of [nn package document](https://github.com/torch/nn/blob/master/README.md). Blank cell indicates not implemented.

## Containers
|module|CPU code/test|GPU code/test|
|----|----|----|
|Sequential|+/-||
|Parallel|||
|Concat|+/-||
|DepthConcat|+/-||
|Bottle|||

## Transfer functions
|module|CPU code/test|GPU code/test|
|----|----|----|
|HardTanh|||
|HardShrink|||
|SoftShrink|||
|SoftMax|||
|SoftMin|||
|SoftPlus|||
|SoftSign|||
|LogSigmoid|||
|LogSoftMax|||
|Sigmoid|||
|Tanh|||
|ReLU|+/-||
|ReLU6|||
|PReLU|||
|RReLU|||
|ELU|||
|LeakyReLU|||
|SpatialSoftMax|||
|AddConstant|||
|MulConstant|+/-||


## Simple layers
###Parameterized Modules
|module|CPU code/test|GPU code/test|
|----|----|----|
|Linear|+/-||
|SparseLinear|||
|BiLinear|||
|PartialLinear|||
|Add|||
|Mul|||
|CMul|||
|Euclidean|||
|WeightedEuclidean|||
|Cosine|||


###Modules that adapt basic Tensor methods 

|module|CPU code/test|GPU code/test|
|----|----|----|
|Copy|||
|Narrow|||
|Replicate|||
|Reshape|+/-||
|View|+/-||
|Contiguous|||
|Select|||
|MaskedSelect|||
|Index|||
|Squeeze|||
|Unsqueeze|||
|Transpose|||

###Modules that adapt mathematical Tensor methods
|module|CPU code/test|GPU code/test|
|----|----|----|
|AddConstant|||
|MulConstant|+/-||
|Max|||
|Min|||
|Mean|||
|Sum|||
|Exp|||
|Log|||
|Abs|||
|Power|||
|Square|||
|Sqrt|+/-||
|Clamp|||
|Normalize|+/-||
|MM|||

###Miscellaneous Modules
|module|CPU code/test|GPU code/test|
|----|----|----|
|BatchNormalization|+/-||
|Identity|||
|Dropout|||
|SpatialDropout|||
|VolumetricDropout|||
|Padding|||
|L1Penalty|||
|GradientReversal|||
|GPU|||
|TemporalDynamicKMaxPooling|||

## Table layers
### table Container Modules encapsulate sub-Modules
|module|CPU code/test|GPU code/test|
|----|----|----|
|ConcatTable|||
|ParallelTable|||
|MapTable|||

### Table Conversion Modules convert between tables and Tensors or tables
|module|CPU code/test|GPU code/test|
|----|----|----|
|SplitTable|||
|JoinTable|||
|MixtureTable|||
|SelectTable|||
|NarrowTable|||
|FlattenTable|||

### Pair Modules compute a measure like distance or similarity from a pair (table) of input Tensors
|module|CPU code/test|GPU code/test|
|----|----|----|
|PairwiseDistance|||
|DotProduct|||
|CosineDistance|||

### CMath Modules perform element-wise operations on a table of Tensors
|module|CPU code/test|GPU code/test|
|----|----|----|
|CAddTable|||
|CSubTable|||
|CMulTable|||
|CDivTable|||
|CMaxTable|||
|CMinTable|||

### Table of Criteria
|module|CPU code/test|GPU code/test|
|----|----|----|
|CriterionTable|||

## Convolution layers
### Temporal Modules
|module|CPU code/test|GPU code/test|
|----|----|----|
|TemporalConvolution|||
|TemporalSubSampling|||
|TemporalMaxPooling|||
|LookupTable|||

### Spatial Modules
|module|CPU code/test|GPU code/test|
|----|----|----|
|SpatialConvolution|+/-||
|SpatialConvolutionMM|+/-||
|SpatialFullConvolution|||
|SpatialDilatedConvolution|||
|SpatialConvolutionLocal|||
|SpatialSubSampling|||
|SpatialMaxPooling|+/-||
|SpatialDilatedMaxPooling|||
|SpatialFractionalMaxPooling|||
|SpatialAveragePooling|+/-||
|SpatialAdaptvieMaxPooling|||
|SpatialMaxUnpooling|||
|SpatialLPPooling|+/-||
|SpatialConvolutionMap|||
|SpatialZeroPadding|||
|SpatialReflectionPadding|+/-||
|SpatialReplicationPadding|||
|SpatialSubtractiveNormalization|||
|SpatialCrossMapLRN|+/-||
|SpatialBatchNormalization|+/-||
|SpatialUpsamplingNearest|||
|SpatialUpsamplingBilinear|||

### Volumetric Modules
|module|CPU code/test|GPU code/test|
|----|----|----|
|VolumetricConvolution|||
|VolumetricFullConvolution|||
|VolumetricDilatedConvolution|||
|VolumetricMaxPooling|||
|VolumetricDilatedMaxPooling|||
|VolumetricAveragePooling|||
|VolumetricMaxUnpooling|||
|VolumetricReplicationPadding|||

# dpnn package
Currently only [Inception](https://github.com/Element-Research/dpnn#nn.Inception) and [Decorator](https://github.com/Element-Research/dpnn#nn.Decorator) module are supported.
