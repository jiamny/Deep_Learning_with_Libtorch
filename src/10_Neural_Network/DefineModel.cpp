
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

/***********************************************************************
构建模型的3种方法

可以使用以下3种方式构建模型：
1，继承nn.Module基类构建自定义模型。
2，使用nn.Sequential按层顺序构建模型。
3，继承nn.Module基类构建模型并辅助应用模型容器进行封装(nn.Sequential,nn.ModuleList,nn.ModuleDict)。
其中 第1种方式最为常见，第2种方式最简单，第3种方式最为灵活也较为复杂。

推荐使用第1种方式构建模型。
***********************************************************************/
// -----------------------------------------
// 一，继承nn.Module基类构建自定义模型
// -----------------------------------------
struct NetImpl : public torch::nn::Module {
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
	torch::nn::Dropout2d dropout{nullptr};
	torch::nn::AdaptiveMaxPool2d adaptive_pool{nullptr};
	torch::nn::Flatten flatten{nullptr};
	torch::nn::Linear linear1{nullptr}, linear2{nullptr};
	torch::nn::ReLU relu{nullptr};
	torch::nn::Sigmoid sigmoid{nullptr};

	NetImpl() {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3));
        pool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        conv2 =torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5));
        pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        dropout = torch::nn::Dropout2d ( torch::nn::Dropout2dOptions(0.1) );
        adaptive_pool = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1,1}));
        flatten = torch::nn::Flatten();
        linear1 = torch::nn::Linear(64,32);
        relu = torch::nn::ReLU();
        linear2 = torch::nn::Linear(32,1);
        sigmoid = torch::nn::Sigmoid();
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("linear1", linear1);
        register_module("linear2", linear2);
	}

    torch::Tensor forward( torch::Tensor x ) {
        x = conv1->forward(x);
        x = pool1->forward(x);
        x = conv2->forward(x);
        x = pool2->forward(x);
        x = dropout->forward(x);
        x = adaptive_pool->forward(x);
        x = flatten->forward(x);
        x = linear1->forward(x.reshape({x.size(1), x.size(0)}));
        x = relu->forward(x);
        x = linear2->forward(x);
        auto y = sigmoid->forward(x);
        return y;
    }
};
TORCH_MODULE(Net);

// ----------------------------------------------
// 三，继承nn.Module基类构建模型并辅助应用模型容器进行封装
//
// 当模型的结构比较复杂时，我们可以应用模型容器(nn.Sequential,nn.ModuleList,nn.ModuleDict)对模型的部分结构进行封装。
// ----------------------------------------------
struct Net2Impl : public torch::nn::Module {
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
	torch::nn::Dropout2d dropout{nullptr};
	torch::nn::AdaptiveMaxPool2d adaptive_pool{nullptr};
	torch::nn::Flatten flatten{nullptr};
	torch::nn::Linear linear1{nullptr}, linear2{nullptr};
	torch::nn::ReLU relu{nullptr};
	torch::nn::Sigmoid sigmoid{nullptr};
	torch::nn::Sequential conv{nullptr}, dense{nullptr};

	Net2Impl() {
		conv = torch::nn::Sequential({
			{"conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3))},
			{"pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))},
			{"conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5))},
			{"pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))},
			{"dropout",  torch::nn::Dropout2d ( torch::nn::Dropout2dOptions(0.1) )},
			{"adaptive_pool", torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1,1}))},
			{"flatten", torch::nn::Flatten()}
		});
		dense = torch::nn::Sequential({
			{"linear1", torch::nn::Linear(64,32)},
			{"relu", torch::nn::ReLU()},
			{"linear2", torch::nn::Linear(32,1)},
			{"sigmoid", torch::nn::Sigmoid()}
		});
		register_module("conv", conv);
		register_module("dense", dense);
	}

    torch::Tensor forward( torch::Tensor x ) {
    	x = conv->forward(x);
    	x = x.reshape({x.size(1), x.size(0)});
    	auto y = dense->forward(x);
        return y;
    }
};
TORCH_MODULE(Net2);

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// 一，继承nn.Module基类构建自定义模型
	auto net = Net();
	net->to(device);
	std::cout << net << '\n';

	auto x = torch::randn({3, 32, 32}).to(device);
	std::cout << x.sizes() << '\n';
	std::cout << net->forward(x) << '\n';

	// -----------------------------------------------
	// 二，使用nn.Sequential按层顺序构建模型
	//
	// 使用nn.Sequential按层顺序构建模型无需定义forward方法。仅仅适合于简单的模型。
	// -----------------------------------------------

	torch::nn::Sequential model ({
		{"conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3))},
		{"pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))},
		{"conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5))},
		{"pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))},
		{"dropout",  torch::nn::Dropout2d ( torch::nn::Dropout2dOptions(0.1) )},
		{"adaptive_pool", torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1,1}))},
		{"flatten", torch::nn::Flatten()},
		{"linear1", torch::nn::Linear(64,32)},
		{"relu", torch::nn::ReLU()},
		{"linear2", torch::nn::Linear(32,1)},
		{"sigmoid", torch::nn::Sigmoid()}
	});

	// 利用OrderedDict

	model->to(device);
	std::cout << model << '\n';

	// ----------------------------------------------
	// 三，继承nn.Module基类构建模型并辅助应用模型容器进行封装
	// ----------------------------------------------
	std::cout << "nn.Sequential作为模型容器\n";
	auto net2 = Net2();
	net2->to(device);
	std::cout << net2 << '\n';

	x = torch::randn({3, 32, 32}).to(device);
	std::cout << x.sizes() << '\n';
	std::cout << net2->forward(x) << '\n';

	std::cout << "Done!\n";
	return 0;
}









