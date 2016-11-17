#include <cpptorch/cpptorch.h>

#include <fstream>
#include <iostream>


template <typename T, GPUFlag F>
class InstanceNormalization : public cpptorch::nn::SpatialBatchNormalization<T, F>
{
public:
    InstanceNormalization(const cpptorch::object_torch *torch_obj, cpptorch::layer_creator<T, F> *creator)
    {
        const cpptorch::object_table *obj_tbl = torch_obj->data_->to_table();
        this->eps_ = *obj_tbl->get("eps");
        this->prev_N_ = -1;
        this->weight_ = creator->read_tensor(obj_tbl->get("weight"));
        this->bias_ = creator->read_tensor(obj_tbl->get("bias"));
    }

    virtual const std::string name() const {
        return "nn.InstanceNormalization";
    }

    virtual cpptorch::Tensor<T, F> forward(const cpptorch::Tensor<T, F> &input) const
    {
        cpptorch::Tensor<T, F> output(true);
        return output;
    }

protected:
    double eps_;
    int prev_N_;
    cpptorch::Tensor<T, F> weight_, bias_;
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
            return std::shared_ptr<cpptorch::nn::Layer<T, F>>(new InstanceNormalization<T, F>(torch_obj, this));
        }
        return nullptr;
    }
};

void test_fast_neural_style(const char *root)
{
    my_layer_creator<float, GPU_None> m;
    std::ifstream fs(std::string(root) + "/candy.t7", std::ios::binary);
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
