#include <cpptorch_cuda.h>

#include <fstream>
#include <algorithm>
#include <iostream>
#include <chrono>


std::shared_ptr<cpptorch::nn::CudaLayer> read_cuda_layer(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_cuda_net(obj.get());
}


cpptorch::CudaTensor read_cuda_tensor(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    assert(fs.good());
    auto obj = cpptorch::load(fs);
    return cpptorch::read_cuda_tensor(obj.get());
}


void test_cuda_layer(const char *data_path, const char *subdir, int count = 1)
{
    auto net = read_cuda_layer(std::string(data_path) + "/" + subdir + "/net.t7");
    cpptorch::CudaTensor x = read_cuda_tensor(std::string(data_path) + "/" + subdir + "/x.t7");
    cpptorch::CudaTensor y = read_cuda_tensor(std::string(data_path) + "/" + subdir + "/y.t7");
    std::cout << *net;

    auto begin = std::chrono::high_resolution_clock::now();
    cpptorch::CudaTensor yy;
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


int main(int argc, char *argv[])
{
    cpptorch::cuda::init();

    //    test_cuda_layer(argv[1], "Inception");
    //    test_cuda_layer(argv[1], "InceptionBig");
    //    test_cuda_layer(argv[1], "Linear");
    //    test_cuda_layer(argv[1], "MulConstant");
    //    test_cuda_layer(argv[1], "Normalize");
    //    test_cuda_layer(argv[1], "Normalize_2d");
    //    test_cuda_layer(argv[1], "Normalize_inf");
    //    test_cuda_layer(argv[1], "ReLU");
    //    test_cuda_layer(argv[1], "Reshape");
    //    test_cuda_layer(argv[1], "Reshape_batch");
    //    test_cuda_layer(argv[1], "SpatialAveragePooling");
    //    test_cuda_layer(argv[1], "SpatialBatchNormalization", 100000);
    //    test_cuda_layer(argv[1], "SpatialConvolution");
    //    test_cuda_layer(argv[1], "SpatialCrossMapLRN");
    //    test_cuda_layer(argv[1], "SpatialMaxPooling");
    //    test_cuda_layer(argv[1], "SpatialReflectionPadding");
    //    test_cuda_layer(argv[1], "Sqrt");
    //    test_cuda_layer(argv[1], "Square");
    //    test_cuda_layer(argv[1], "View");
    test_cuda_layer(argv[1], "_face", 10);

    //test_fast_neural_style(argv[1], "candy");
    cpptorch::cuda::free();

#ifdef _WIN64
    _CrtDumpMemoryLeaks();
#endif
    return 0;
}
