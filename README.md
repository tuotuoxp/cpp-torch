# Introduction

cpp-torch is a C++ library, implemented as a wrapper around [torch](https://github.com/torch) **C libraries** (not lua libraries).

Using this library, you can:

- load torch data tensor from `.t7` file
- load torch network model from `.t7` file
- feed data into model, perform forward pass and get output

**All in C++, without touching lua.**

Pretty handy when you want to deploy an off-the-shelf torch model.

# Install
Check our [install script](install.md) in Linux, Windows and MacOS.

# Get started
A simple load-and-fire routine looks like this in C++:

```c++
// read input tensor
std::ifstream fs_input("input_tensor.t7", std::ios::binary);
auto obj_input = cpptorch::load(fs_input);
auto input = cpptorch::read_tensor<TensorFloat>(obj_input.get());
std::cout << input << std::endl;

// read network
std::ifstream fs_net("net.t7", std::ios::binary);
auto obj_net = cpptorch::load(fs_net);
auto net = cpptorch::read_net<TensorFloat>(obj_net.get());
std::cout << *net << std::endl;

// forward
auto output = net->forward(input);
std::cout << output << std::endl;
```

We also provides an [example script]() to test the famous [CMU OpenFace](https://github.com/cmusatyalab/openface) model. This network transfers a 3 * 96 * 96 face image into a 128 * 1 feature vector, representing the identity of the person.

# Performance
This wrapper is **about 2x faster** than torch's lua implementation in CPU mode.

|model|torch|cpp-torch|
|----|----|----|
|nn4.small2|?|?|


# Progress
Currently, this library supports forward pass in CPU mode of
- some modules in [nn package](https://github.com/torch/nn)
- related functions in [torch7 package](https://github.com/torch/torch7)
- a few modules in [dpnn package](https://github.com/Element-Research/dpnn).

Check [this list](progress.md) to see supported modules.


# Next step work
- Foward pass of other modules in nn package
- Wrapper for GPU mode

# FAQ
-- How can I train my own model with this wrapper?
-- We don't support backward functions, so training is impossible. Use the original torch.