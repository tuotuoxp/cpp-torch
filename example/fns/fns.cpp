#include <cpptorch/cpptorch.h>

#include <fstream>
#include <iostream>


template <typename T, GPUFlag F>
class InstanceNormalization : public cpptorch::nn::Layer<T, F>
{
public:
    InstanceNormalization(const cpptorch::object_torch *torch_obj)
    {

    }

    virtual const std::string name() const {
        return "nn.InstanceNormalization";
    }

    virtual cpptorch::Tensor<T, F> forward(const cpptorch::Tensor<T, F> &input) const
    {
        cpptorch::Tensor<T, F> output(true);
        return output;
    }
};


template <typename T, GPUFlag F>
class my_layer_creator : public cpptorch::layer_creator<T, F>
{
public:
    std::vector<std::string> register_layers()
    {
        return { "nn.InstanceNormalization" };
    }
    std::shared_ptr<cpptorch::nn::Layer<T, F>> create_layer(const std::string &layer_name, const cpptorch::object_torch *torch_obj)
    {
        if (layer_name == "nn.InstanceNormalization")
        {
            return std::shared_ptr<cpptorch::nn::Layer<T, F>>(new InstanceNormalization<T, F>(torch_obj));
        }
        return nullptr;
    }
};

void test_fast_neural_style(const char *root)
{
    my_layer_creator<float, GPU_None> m;
    std::ifstream fs(std::string(root) + "/_fast_neural_style/candy.t7", std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    auto net = cpptorch::read_net<float>(obj->to_table()->get("model"), &m);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



int main(int argc, char *argv[])
{
    cpptorch::allocator::init();
    test_fast_neural_style(argv[1]);
    cpptorch::allocator::cleanup();

#ifdef _WIN64
    _CrtDumpMemoryLeaks();
#endif
    return 0;
}
