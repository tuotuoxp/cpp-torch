#include <cpptorch.h>

#include <fstream>
#include <algorithm>
#include <iostream>
#include <chrono>


template<typename T>
std::shared_ptr<cpptorch::nn::Layer<T>> read_layer(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_net<T>(obj.get());
}

template<typename T>
cpptorch::Tensor<T> read_tensor(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_tensor<T>(obj.get());
}

void test_index(const char *data_path)
{
    cpptorch::Tensor<float> x = read_tensor<float>(std::string(data_path) + "/_index/x.t7");
    cpptorch::Tensor<float> y1 = read_tensor<float>(std::string(data_path) + "/_index/y1.t7");
    cpptorch::Tensor<float> y2 = read_tensor<float>(std::string(data_path) + "/_index/y2.t7");
    std::ifstream fs(std::string(data_path) + "/_index/y3.t7", std::ios::binary);
    float y3 = *cpptorch::load(fs);

    std::cout << x[1] - y1 << std::endl;
    std::cout << x[0][1][3] - y2 << std::endl;
    std::cout << x[{0, 1, 3}] - y2 << std::endl;
    std::cout << (float)x[{1, 0, 4, 1}] - y3 << std::endl;
}


void test_layer(const char *data_path, const char *subdir, int count = 1)
{
    auto net = read_layer<float>(std::string(data_path) + "/" + subdir + "/net.t7");
    cpptorch::Tensor<float> x = read_tensor<float>(std::string(data_path) + "/" + subdir + "/x.t7");
    cpptorch::Tensor<float> y = read_tensor<float>(std::string(data_path) + "/" + subdir + "/y.t7");
    std::cout << *net;

    auto begin = std::chrono::high_resolution_clock::now();
    cpptorch::Tensor<float> yy;
    for (int i = 0; i < count; i++)
    {
        yy = net->forward(x);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto sub = cpptorch::abs(y - yy);
    if (sub.minall() > 1e-05 || sub.maxall() > 1e-05)
    {
        std::cout << "----------------------- FAILED!!!!!!!!";
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
    std::cout << "================================================" << std::endl;
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


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

//    test_index(argv[1]);
//    test_layer(argv[1], "Inception");
//    test_layer(argv[1], "InceptionBig");
//    test_layer(argv[1], "Linear");
//    test_layer(argv[1], "MulConstant");
//    test_layer(argv[1], "Normalize");
//    test_layer(argv[1], "Normalize_2d");
//    test_layer(argv[1], "Normalize_inf");
//    test_layer(argv[1], "ReLU");
//    test_layer(argv[1], "Reshape");
//    test_layer(argv[1], "Reshape_batch");
//    test_layer(argv[1], "SpatialAveragePooling");
//    test_layer(argv[1], "SpatialBatchNormalization", 100000);
//    test_layer(argv[1], "SpatialConvolution");
//    test_layer(argv[1], "SpatialCrossMapLRN");
//    test_layer(argv[1], "SpatialMaxPooling");
//    test_layer(argv[1], "SpatialReflectionPadding");
//    test_layer(argv[1], "Sqrt");
//    test_layer(argv[1], "Square");
//    test_layer(argv[1], "View");
//    test_layer(argv[1], "_face", 10);

    test_fast_neural_style(argv[1]);
    cpptorch::allocator::cleanup();

#ifdef _WIN64
    _CrtDumpMemoryLeaks();
#endif
    return 0;
}
