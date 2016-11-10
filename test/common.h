#include <stdio.h>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <chrono>

#include <cpptorch.h>


static int optind = 0, opterr = 0;
static char *optarg = nullptr;
static int run_times = 1;
static char *input = nullptr;
static char *output = nullptr;
static char *network = nullptr;
static bool use_allocator = false;


int my_getopt(int argc, char *argv[], char *optstring)
{
	static char *next = nullptr;
	if (optind == 0)
		next = nullptr;

	optarg = nullptr;

	if (next == nullptr || *next == '\0')
	{
		if (optind == 0)
			optind++;

		if (optind >= argc || argv[optind][0] != '-' || argv[optind][1] == '\0')
		{
			optarg = nullptr;
			if (optind < argc)
				optarg = argv[optind];
			return -1;
		}

		if (strcmp(argv[optind], "--") == 0)
		{
			optind++;
			optarg = nullptr;
			if (optind < argc)
				optarg = argv[optind];
			return -1;
		}

		next = argv[optind];
		next++;		// skip past -
		optind++;
	}

	char c = *next++;
	char *cp = strchr(optstring, c);

	if (cp == nullptr || c == ':')
		return '?';

	cp++;
	if (*cp == ':')
	{
		if (*next != '\0')
		{
			optarg = next;
			next = nullptr;
		}
		else if (optind < argc)
		{
			optarg = argv[optind];
			optind++;
		}
		else
		{
			return '?';
		}
	}

	return c;
}


void usage()
{
    std::cerr << std::endl
        << "Usage: -i [input tensor] -o [output tensor] -n [network]" << std::endl
        << std::endl
        << "Additional options:" << std::endl
        << "       -a         use optimized allocator (only for cpu)" << std::endl
        << "       -t [n]     run forward operation n-times" << std::endl
        << std::endl;
}


int process_args(int argc, char *argv[])
{
    int c = 0;
    while ((c = my_getopt(argc, argv, "ai:o:n:t:")) >= 0)
    {
        switch (c)
        {
        case 'a':
            use_allocator = true;
            break;
        case 'i':
            input = optarg;
            break;
        case 'o':
            output = optarg;
            break;
        case 'n':
            network = optarg;
            break;
        case 't':
            run_times = atoi(optarg);
            break;
        case '?':
            usage();
            return 10;
        }
    }

    if (!input || !output || !network)
    {
        usage();
        return 10;
    }

    return 0;
}


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> read_tensor_template(const cpptorch::object *obj);

template<typename T, GPUFlag F>
std::shared_ptr<cpptorch::nn::Layer<T, F>> read_net_template(const cpptorch::object *obj);


template<typename T, GPUFlag F>
cpptorch::Tensor<T, F> read_tensor(const char *tensor)
{
    std::ifstream fs(tensor, std::ios::binary);
    if (!fs.good())
    {
        std::cerr << "Cannot read tensor from " << tensor << std::endl;
        exit(100);
    }
    auto obj = cpptorch::load(fs);
    return read_tensor_template<T, F>(obj.get());
}

template<typename T, GPUFlag F>
int test_layer()
{
    std::ifstream fs(network, std::ios::binary);
    if (!fs.good())
    {
        std::cerr << "Cannot read network from " << network << std::endl;
        exit(101);
    }
    auto obj = cpptorch::load(fs);
    auto net = read_net_template<T, F>(obj.get());
    cpptorch::Tensor<T, F> x = read_tensor<T, F>(input);
    cpptorch::Tensor<T, F> y = read_tensor<T, F>(output);

    auto begin = std::chrono::high_resolution_clock::now();
    cpptorch::Tensor<T, F> yy;
    for (int i = 0; i < run_times; i++)
    {
        yy = net->forward(x);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto sub = cpptorch::abs(y - yy);
    if (sub.minall() > 1e-05 || sub.maxall() > 1e-05)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    return 0;
}
