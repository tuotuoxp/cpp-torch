#include <cpptorch/cpptorch.h>

#include <fstream>
#include <iostream>


template<class T>
cpptorch::Tensor<T> read_tensor(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_tensor<T>(obj.get());
}

template<class T>
std::shared_ptr<cpptorch::nn::Layer<T>> read_layer(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_net<T>(obj.get());
}

int main(int argc, char *argv[])
{
    cpptorch::Tensor<float> x = read_tensor<float>("x.t7");
    cpptorch::Tensor<float> y = read_tensor<float>("y.t7");
    auto net = read_layer<float>("net.t7");
    std::cout << *net << std::endl;

    auto yy = net->forward(x);
    std::cout << y << std::endl;
    std::cout << yy << std::endl;

    return 0;
}
