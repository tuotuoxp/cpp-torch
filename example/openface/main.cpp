#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cpptorch/cpptorch.h>

const int img_dim = 96;

int main(int argc, char** argv)
{
    // 1. load 96*96 RGB image to OpenCV Mat
    cv::Mat image = cv::imread("face.jpg");
    if (image.channels() != 3 || image.rows != img_dim || image.cols != img_dim)
    {
        std::cerr << "invalid size" << image.channels() << " " << image.rows << " " << image.cols << " " << std::endl;
        return 0;
    }

    // 2. create input tensor from CV Mat
    cpptorch::Tensor<float> input;
    input.create();
    input.resize({1, 3, image.rows, image.cols});

    const unsigned char *img = image.ptr(0);
    float *ten = input.data();
    for (size_t c = 0; c < 3; c++)
    {
        for (size_t p = 0; p < img_dim * img_dim; p++)
        {
            ten[c * img_dim * img_dim + p] = (float)img[p * 3 + 2 - c] / 255;    // normalize  to [0,1]
        }
    }

    // 3. load openface network
    std::ifstream fs_net(std::string("nn4.small2.v1.t7"), std::ios::binary);
    assert(fs_net.good());   // http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7
    auto obj_t = cpptorch::load(fs_net);
    std::shared_ptr<cpptorch::nn::Layer<float>> net = cpptorch::read_net<float>(obj_t.get());

    // 4. foward
    cpptorch::Tensor<float> output = net->forward(input);
    
    // 5. print 1*128 output
    ten = output.data();
    for (int i = 0; i < 128; i++)
    {
        std::cout << ten[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
