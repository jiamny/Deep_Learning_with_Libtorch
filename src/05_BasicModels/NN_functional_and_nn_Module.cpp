#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <list>
#include <algorithm>
#include <unistd.h>
#include <iomanip>

/***************************************************************************
其中nn.functional(一般引入后改名为F)有各种功能组件的函数实现。例如：

(激活函数)
* F.relu
* F.sigmoid
* F.tanh
* F.softmax

(模型层)
* F.linear
* F.conv2d
* F.max_pool2d
* F.dropout2d
* F.embedding

(损失函数)
* F.binary_cross_entropy
* F.mse_loss
* F.cross_entropy

为了便于对参数进行管理，一般通过继承 nn.Module 转换成为类的实现形式，并直接封装在 nn 模块下。例如：

(激活函数)
* nn.ReLU
* nn.Sigmoid
* nn.Tanh
* nn.Softmax

(模型层)
* nn.Linear
* nn.Conv2d
* nn.MaxPool2d
* nn.Dropout2d
* nn.Embedding

(损失函数)
* nn.BCELoss
* nn.MSELoss
* nn.CrossEntropyLoss
*************************************************************************************/

struct Net : public torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::Sequential conv{nullptr}, dense{nullptr};

    Net() {
        embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(10000, 3).padding_idx(1));
        conv = torch::nn::Sequential(
        		register_module("conv_1", torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 16, 5))),
				register_module("pool_1", torch::nn::MaxPool1d(2)),
				register_module("relu_1", torch::nn::ReLU()),
				register_module("conv_2", torch::nn::Conv1d(torch::nn::Conv1dOptions(16, 128, 2))),
				register_module("pool_2", torch::nn::MaxPool1d(2)),
				register_module("relu_2", torch::nn::ReLU()));

        dense = torch::nn::Sequential(
        		register_module("flatten", torch::nn::Flatten()),
				register_module("linear", torch::nn::Linear(6144,1)),
				register_module("sigmoid", torch::nn::Sigmoid()));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = embedding->forward(x).transpose(1,2);
        x = conv->forward(x);
        auto y = dense->forward(x);
        return y;
    }
};


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	auto w = torch::randn({2,2}).to(device);
	std::cout << "w:\n" << w << "\n";
	std::cout << "w.requires_grad: " << w.requires_grad() << "\n";

	// nn.Parameter 具有 requires_grad = True 属性
	auto module = torch::nn::Module();
	module.to(device);
	module.register_parameter("w", w);
	std::cout << "module.w.requires_grad: " << module.named_parameters().find("w")->requires_grad() << "\n";

	auto net = Net();

	size_t i = 0;
	for(auto& child : net.children() ) {
	    i+=1;
	    std::cout << *child << "\n";
	}
	std::cout << "child number: " << i << "\n";

	return 0;
}



