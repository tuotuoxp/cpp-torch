# Test
We provide some [cases](/test/cases) to validate our cpp modules.
Given input tensor and network model, each test case compares the output tensor under original lua module and our cpp module.

## Build test
During 'Install torch wrapper' step, add DBUILD_TESTS=ON to cmake setting.

For CPU version:
```
cmake -DBUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ..
```

For GPU version:
```
cmake -DBUILD_TESTS=ON -DWITH_CUDA=1 -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ..
```

## Use test
Go to `cpp-torch/build/` and run 'ctest' in command.
You will get something like this:
```
92% tests passed, 2 tests failed out of 24

Total Test time (real) =  11.16 sec

The following tests FAILED:
          7 - cpptorch_ConvolutionLayers_Spatial_SpatialBatchNormalization_2.lua (Failed)
         14 - cpptorch_dpnn_Decorator_1.lua (Failed)
```

## Write new test case
Each `.lua` file under [test/cases](/test/cases) is a test case.
Please follow torch's [nn package document](https://github.com/torch/nn/blob/master/README.md) to organize cases as follows:

├─Containers
├─ConvolutionLayers
│ ├─Spatial
│ ├─Temporal
│ └─Volumetric
├─dpnn
├─SimpleLayers
│ ├─BasicTensor
│ ├─MathTensor
│ ├─Miscellaneous
│ └─Parameterized
├─TableLayers
│ ├─CMath
│ ├─Container
│ ├─Conversion
│ ├─Criteria
│ └─Pair
└─TransferFunctions

# Write torch layer?

# Write self-defined layer?
