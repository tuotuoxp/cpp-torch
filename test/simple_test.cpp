#include "../src/nn.h"
#include "../src/extractor.h"
#include "../src/builder.h"
#include "../src/builder.h.inl"

#include <fstream>
#include <algorithm>
#include <iostream>
#include <chrono>


template<class TTensor, template<typename> class T>
std::shared_ptr<T<TTensor>> read_layer(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = model_extractor().read_object(fs);
    model_builder<TTensor> mb;
    
    return std::static_pointer_cast<T<TTensor>>(mb.build_layer(obj.get()));
}

template<class TTensor>
nn::Tensor<TTensor> read_tensor(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = model_extractor().read_object(fs);
    model_builder<TTensor> mb;
    return mb.build_tensor(obj.get());
}

void test_index(const char *data_path)
{
    nn::Tensor<TensorFloat> x = read_tensor<TensorFloat>(std::string(data_path) + "/_index/x.t7");
    nn::Tensor<TensorFloat> y1 = read_tensor<TensorFloat>(std::string(data_path) + "/_index/y1.t7");
    nn::Tensor<TensorFloat> y2 = read_tensor<TensorFloat>(std::string(data_path) + "/_index/y2.t7");
    std::ifstream fs(std::string(data_path) + "/_index/y3.t7", std::ios::binary);
    float y3 = *model_extractor().read_object(fs);

    std::cout << x[1] - y1 << std::endl;
    std::cout << x[0][1][3] - y2 << std::endl;
    std::cout << x[{0, 1, 3}] - y2 << std::endl;
    std::cout << (float)x[{1, 0, 4, 1}] - y3 << std::endl;
}


void test_layer(const char *data_path, const char *subdir)
{
    nn::Tensor<TensorFloat> x = read_tensor<TensorFloat>(std::string(data_path) + "/" + subdir + "/x.t7");
    nn::Tensor<TensorFloat> y = read_tensor<TensorFloat>(std::string(data_path) + "/" + subdir + "/y.t7");
    auto net = read_layer<TensorFloat, nn::Layer>(std::string(data_path) + "/" + subdir + "/net.t7");
    std::cout << *net;

    auto begin = std::chrono::high_resolution_clock::now();
    auto yy = net->forward(x);
    auto end = std::chrono::high_resolution_clock::now();
    auto sub = nn::abs(y - yy);
    if (sub.minall() > 1e-05 || sub.maxall() > 1e-05)
    {
        std::cout << "----------------------- FAILED!!!!!!!!";
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
    std::cout << "================================================" << std::endl;
}


void test_temp(const char *data_path, const char *subdir)
{
    nn::Tensor<TensorFloat> x = read_tensor<TensorFloat>(std::string(data_path) + "/" + subdir + "/x.t7");
    nn::Tensor<TensorFloat> y = read_tensor<TensorFloat>(std::string(data_path) + "/" + subdir + "/y_sub.t7");
    auto net = read_layer<TensorFloat, nn::Layer>(std::string(data_path) + "/" + subdir + "/net_sub.t7");
    std::cout << *net;

    auto begin = std::chrono::high_resolution_clock::now();
    auto yy = net->forward(x);
    auto end = std::chrono::high_resolution_clock::now();
    auto sub = nn::abs(y - yy);
    if (sub.minall() > 1e-05 || sub.maxall() > 1e-05)
    {
        std::cout << "----------------------- FAILED!!!!!!!!  (" << sub.minall() << ", " << sub.maxall() << ")" << std::endl;
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
    std::cout << "================================================" << std::endl;
}


int main(int argc, char *argv[])
{
    //test_index(argv[1]);
    test_layer(argv[1], "_face");
//     test_layer(argv[1], "Inception");
//     test_layer(argv[1], "InceptionBig");
//     test_layer(argv[1], "Linear");
//     test_layer(argv[1], "MulConstant");
//     test_layer(argv[1], "Normalize");
//     test_layer(argv[1], "Normalize_2d");
//     test_layer(argv[1], "Normalize_inf");
//     test_layer(argv[1], "ReLU");
//     test_layer(argv[1], "Reshape");
//     test_layer(argv[1], "Reshape_batch");
//     test_layer(argv[1], "SpatialAveragePooling");
//     test_layer(argv[1], "SpatialBatchNormalization");
//     test_layer(argv[1], "SpatialConvolution");
//     test_layer(argv[1], "SpatialCrossMapLRN");
//     test_layer(argv[1], "SpatialMaxPooling");
//     test_layer(argv[1], "Sqrt");
//     test_layer(argv[1], "Square");
//     test_layer(argv[1], "View");

#ifdef _WIN64
    _CrtDumpMemoryLeaks();
#endif
    return 0;
}
