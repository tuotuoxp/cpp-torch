#include <cpptorch/cpptorch.h>

#include <fstream>
#include <iostream>


template<class TTensor>
cpptorch::Tensor<TTensor> read_tensor(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_tensor<TTensor>(obj.get());
}

template<class TTensor>
std::shared_ptr<cpptorch::nn::Layer<TTensor>> read_layer(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_net<TTensor>(obj.get());
}

int main(int argc, char *argv[])
{
    cpptorch::Tensor<TensorFloat> x = read_tensor<TensorFloat>("x.t7");
    cpptorch::Tensor<TensorFloat> y = read_tensor<TensorFloat>("y.t7");
    auto net = read_layer<TensorFloat>("net.t7");
    std::cout << *net << std::endl;

    auto yy = net->forward(x);
    std::cout << y << std::endl;
    std::cout << yy << std::endl;

    return 0;
}
 