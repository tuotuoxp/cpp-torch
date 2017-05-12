# Introduction

cpp-torch is a C++ library, implemented as a wrapper around [torch](https://github.com/torch) **C libraries** (not lua libraries).

Using this library, you can:

- load torch data tensor from `.t7` file
- load torch network model from `.t7` file
- feed data into model, perform forward pass and get output

**All in C++, without touching lua.**

Pretty handy when you want to deploy an off-the-shelf torch model.

# Install
Check our install script for ~~[Linux](install_linux.md)~~, [Windows](install_windows.md)(*TODO*) and [MacOS](install_mac.md)(*TODO*).

## Install cpp-torch on Linux
The following commands install our C++ wrapper: cpp-torch in default torch install dir:

``` bash
git clone https://github.com/yytdfc/cpp-torch
cd cpp-torch
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/torch/install -DCMAKE_PREFIX_PATH=~/torch/install ..
# For GPU version, set -DBUILD_CUDA=ON:
cmake  -DBUILD_CUDA=ON -DCMAKE_INSTALL_PREFIX=~/torch/install -DCMAKE_PREFIX_PATH=~/torch/install ..
make
make install
cd ..
```

# Get started
The following code loads a float tensor and a float network from file, and forwards the tensor into the network:
```c++
// read input tensor
std::ifstream fs_input("input_tensor.t7", std::ios::binary);
auto obj_input = cpptorch::load(fs_input);
auto input = cpptorch::read_tensor<float>(obj_input.get());     // load float tensor

// read network
std::ifstream fs_net("net.t7", std::ios::binary);
auto obj_net = cpptorch::load(fs_net);
auto net = cpptorch::read_net<float>(obj_net.get());          // load float network

// forward
auto output = net->forward(input);

// display
std::cout << input << std::endl;
std::cout << *net << std::endl;
std::cout << output << std::endl;
```

If tensor and network type is double, change the template type accordingly:
```c++
auto input = cpptorch::read_tensor<double>(obj_input.get());     // load double tensor
auto net = cpptorch::read_tensor<double>(obj_net.get());     // load double network
```

To use GPU, use read_cuda_tensor() function:
```c++
auto input = cpptorch::read_cuda_tensor(obj_input.get());     // load cuda tensor
auto net = cpptorch::read_cuda_net(obj_net.get());          // load cuda network
```

We also provides an [example script](example/openface/main.cpp) to test the famous [CMU OpenFace](https://github.com/cmusatyalab/openface) model. This example transfers a 3 * 96 * 96 face image(OpenCV Mat) into cpp-torch tensor, forwards it through the network, receives a 128 * 1 identity feature tensor, and print the result.

# Progress
Currently, this library supports forward pass of
- some modules in [nn package](https://github.com/torch/nn)
- related functions in [torch7 package](https://github.com/torch/torch7)
- a few modules in [dpnn package](https://github.com/Element-Research/dpnn).

Check [this list](progress.md) to see supported modules.

You are more than welcome to add new modules to cpp-torch. Please check our [developer guide](developer_guide.md).

# FAQ
-- How can I train my own model with this wrapper?

-- We don't support backward functions, so training is impossible. Use the original torch.
