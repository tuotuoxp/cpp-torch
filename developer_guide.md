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
Go to 'cpp-torch/build/' and run 'ctest' in command.
You will get something like this:
```
92% tests passed, 2 tests failed out of 24

Total Test time (real) =  11.16 sec

The following tests FAILED:
          7 - cpptorch_ConvolutionLayers_Spatial_SpatialBatchNormalization_2.lua (Failed)
         14 - cpptorch_dpnn_Decorator_1.lua (Failed)
```

## Write new test case


# Write write torch layer?

# How to write self-defined layer?
