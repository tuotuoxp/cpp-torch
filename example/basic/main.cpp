#include <cpptorch/nn.h>
#include <cpptorch/builder.h>

#include <fstream>
#include <iostream>


template<class TTensor>
nn::Tensor<TTensor> read_tensor(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = model_extractor().read_object(fs);
    return model_builder<TTensor>().build_tensor(obj.get());
}

template<class TTensor, template<typename> class T>
std::shared_ptr<T<TTensor>> read_layer(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = model_extractor().read_object(fs);
    return std::static_pointer_cast<T<TTensor>>(model_builder<TTensor>().build_layer(obj.get()));
}

int main(int argc, char *argv[])
{
    nn::Tensor<TensorFloat> x = read_tensor<TensorFloat>("x.t7");
    nn::Tensor<TensorFloat> y = read_tensor<TensorFloat>("y.t7");
    auto net = read_layer<TensorFloat, nn::Layer>("net.t7");
    std::cout << *net << std::endl;

    auto yy = net->forward(x);
    std::cout << y << std::endl;
    std::cout << yy << std::endl;

    return 0;
}
 