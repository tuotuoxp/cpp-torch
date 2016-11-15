We recommend you read through this document before you develope something new.
- [Use test]( #Use test )
- [Write new test cases](#Write new test case)

# Use test
We provide some [cases](/test/cases) to validate our cpp modules.

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

## Run test
Go to `cpp-torch/build/` and run 'ctest' in command.
You will get something like this:
```
92% tests passed, 2 tests failed out of 24

Total Test time (real) =  11.16 sec

The following tests FAILED:
          7 - cpptorch_ConvolutionLayers_Spatial_SpatialBatchNormalization_2.lua (Failed)
         14 - cpptorch_dpnn_Decorator_1.lua (Failed)
```

# Write new test case
## Organization
Each `.lua` file under [test/cases](/test/cases) is a test case.
Please follow torch's [nn package document](https://github.com/torch/nn/blob/master/README.md) to organize cases as follows:

```
├─ Containers
├─ ConvolutionLayers
│  ├─ Spatial
│  ├─ Temporal
│  └─ Volumetric
├─ dpnn
├─ SimpleLayers
│  ├─ BasicTensor
│  ├─ MathTensor
│  ├─ Miscellaneous
│  └─ Parameterized
├─ TableLayers
│  ├─ CMath
│  ├─ Container
│  ├─ Conversion
│  ├─ Criteria
│  └─ Pair
└─ TransferFunctions
```
A single module can have multiple test cases, e.g.: `Linear_1.lua`, `Linear_2.lua`...

## Write a test case
Check out this test cases for `nn.RELU`:
```lua
require 'torch'
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')
-- create input, outpu and net
input = torch.Tensor(6,3,128,128)
net = nn.ReLU()
net:evaluate()
output = net:forward(input)
-- save
torch.save(arg[1], input)
torch.save(arg[2], output)
net:clearState()
torch.save(arg[3], net)
```
Test case saves input, output and net to destination denoted by the 3 command line args.

Remember to re-build cpp-torch after you add new test cases.

## So what's happening?
When you run `ctest` under `/build`, it performs the following actions:
- parse all `.lua` file under `/test/cases` folder.
- run each `.lua` file in torch, save input tensor, output tensor and net as `.t7` file.
- feed input tensor into net in cpp-torch, compare output tensor with saved output tensor.

# Write torch layer(TODO)

# Write self-defined layer(TODO)
