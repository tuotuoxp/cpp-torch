This list shows current supported modules and test status.

Our purpose is to facilitate the **testing procedure**. So backward functions won't be supported in a foresable future.

# nn package
We follow the structure of [nn package document](https://github.com/torch/nn/blob/master/README.md). Blank cell indicates not implemented.

## Containers
|module|code|CPU test|GPU test|
|----|----|----|----|
|Sequential|:heavy_check_mark:|||
|Parallel||||
|Concat|:heavy_check_mark:|||
|DepthConcat|:heavy_check_mark:|||
|Bottle|||||

## Transfer functions
|module|code|CPU test|GPU test|
|----|----|----|----|
|HardTanh||||
|HardShrink||||
|SoftShrink||||
|SoftMax||||
|SoftMin||||
|SoftPlus||||
|SoftSign||||
|LogSigmoid||||
|LogSoftMax|:heavy_check_mark:|||
|Sigmoid||||
|Tanh||||
|ReLU|:heavy_check_mark:|||
|ReLU6||||
|PReLU||||
|RReLU||||
|ELU||||
|LeakyReLU||||
|SpatialSoftMax||||
|AddConstant||||
|MulConstant|:heavy_check_mark:|||


## Simple layers
### Parameterized Modules
|module|code|CPU test|GPU test|
|----|----|----|----|
|Linear|:heavy_check_mark:|||
|SparseLinear||||
|BiLinear||||
|PartialLinear||||
|Add|:heavy_check_mark:|||
|Mul||||
|CMul||||
|Euclidean||||
|WeightedEuclidean||||
|Cosine||||


### Modules that adapt basic Tensor methods

|module|code|CPU test|GPU test|
|----|----|----|----|
|Copy||||
|Narrow||||
|Replicate||||
|Reshape|:heavy_check_mark:|||
|View|:heavy_check_mark:|||
|Contiguous||||
|Select||||
|MaskedSelect||||
|Index||||
|Squeeze||||
|Unsqueeze||||
|Transpose||||

### Modules that adapt mathematical Tensor methods
|module|code|CPU test|GPU test|
|----|----|----|----|
|AddConstant||||
|MulConstant|:heavy_check_mark:|||
|Max||||
|Min||||
|Mean||||
|Sum||||
|Exp||||
|Log||||
|Abs||||
|Power||||
|Square||||
|Sqrt|:heavy_check_mark:|||
|Clamp||||
|Normalize|:heavy_check_mark:|||
|MM||||

### Miscellaneous Modules
|module|code|CPU test|GPU test|
|----|----|----|----|
|BatchNormalization|:heavy_check_mark:|||
|Identity||||
|Dropout||||
|SpatialDropout||||
|VolumetricDropout||||
|Padding||||
|L1Penalty||||
|GradientReversal||||
|GPU||||
|TemporalDynamicKMaxPooling||||

## Table layers
### table Container Modules encapsulate sub-Modules
|module|code|CPU test|GPU test|
|----|----|----|----|
|ConcatTable||||
|ParallelTable||||
|MapTable||||

### Table Conversion Modules convert between tables and Tensors or tables
|module|code|CPU test|GPU test|
|----|----|----|----|
|SplitTable||||
|JoinTable||||
|MixtureTable||||
|SelectTable||||
|NarrowTable||||
|FlattenTable||||

### Pair Modules compute a measure like distance or similarity from a pair (table) of input Tensors
|module|code|CPU test|GPU test|
|----|----|----|----|
|PairwiseDistance||||
|DotProduct||||
|CosineDistance||||

### CMath Modules perform element-wise operations on a table of Tensors
|module|code|CPU test|GPU test|
|----|----|----|----|
|CAddTable||||
|CSubTable||||
|CMulTable||||
|CDivTable||||
|CMaxTable||||
|CMinTable||||

### Table of Criteria
|module|code|CPU test|GPU test|
|----|----|----|----|
|CriterionTable||||

## Convolution layers
### Temporal Modules
|module|code|CPU test|GPU test|
|----|----|----|----|
|TemporalConvolution||||
|TemporalSubSampling||||
|TemporalMaxPooling||||
|LookupTable||||

### Spatial Modules
|module|code|CPU test|GPU test|
|----|----|----|----|
|SpatialConvolution|:heavy_check_mark:|||
|SpatialConvolutionMM|:heavy_check_mark:|||
|SpatialFullConvolution||||
|SpatialDilatedConvolution||||
|SpatialConvolutionLocal||||
|SpatialSubSampling||||
|SpatialMaxPooling|:heavy_check_mark:|||
|SpatialDilatedMaxPooling||||
|SpatialFractionalMaxPooling||||
|SpatialAveragePooling|:heavy_check_mark:|||
|SpatialAdaptvieMaxPooling||||
|SpatialMaxUnpooling||||
|SpatialLPPooling|:heavy_check_mark:|||
|SpatialConvolutionMap||||
|SpatialZeroPadding||||
|SpatialReflectionPadding|:heavy_check_mark:|||
|SpatialReplicationPadding||||
|SpatialSubtractiveNormalization||||
|SpatialCrossMapLRN|:heavy_check_mark:|||
|SpatialBatchNormalization|:heavy_check_mark:|||
|SpatialUpsamplingNearest||||
|SpatialUpsamplingBilinear||||

### Volumetric Modules
|module|code|CPU test|GPU test|
|----|----|----|----|
|VolumetricConvolution||||
|VolumetricFullConvolution||||
|VolumetricDilatedConvolution||||
|VolumetricMaxPooling||||
|VolumetricDilatedMaxPooling||||
|VolumetricAveragePooling||||
|VolumetricMaxUnpooling||||
|VolumetricReplicationPadding||||

# dpnn package
Currently only [Inception](https://github.com/Element-Research/dpnn#nn.Inception) and [Decorator](https://github.com/Element-Research/dpnn#nn.Decorator) module are supported.
