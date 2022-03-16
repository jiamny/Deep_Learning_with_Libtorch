#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

// Neural Network
struct MyFirstNetworkImpl : torch::nn::Module {
private:
	torch::nn::Linear layer1{nullptr}, layer2{nullptr};
public:
	MyFirstNetworkImpl( int64_t input_size, int64_t hidden_size, int64_t output_size) {
        this->layer1 = torch::nn::Linear(input_size,hidden_size);
        this->layer2 = torch::nn::Linear(hidden_size,output_size);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
	}

	torch::Tensor forward(torch::Tensor input) {
        auto out = layer1->forward(input);
        out = torch::relu(out);
        out = layer2->forward(out);
        return out;
	}
};

TORCH_MODULE(MyFirstNetwork);


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	torch::manual_seed(1000);

	auto xint=torch::ones({2,3}, torch::TensorOptions(torch::kLong));
	std::cout << xint.options() << std::endl;

	auto x = torch::tensor({{1,2,3},{4,5,6}}, torch::TensorOptions(torch::kLong));

	std::cout <<  "x = \n" << x << std::endl;
	std::cout <<  "shape = " << x.sizes() << std::endl;
	std::cout <<  "view(-1) + " << x.view(-1) << std::endl;
	std::cout <<  "view({3,2}) + " << x.view({3,2}) << std::endl;
	std::cout <<  "view({6,1}) + " << x.view({6,1}) << std::endl;

	// Fundamental blocks of Neural Network
	std::cout << "\nFundamental blocks of Neural Network" << std::endl;
	torch::nn::Linear linear_layer = torch::nn::Linear(torch::nn::LinearOptions(5, 3).bias(true));
	auto inp = torch::autograd::Variable(torch::randn({1,5}, dtype_option));
	std::cout << linear_layer->forward(inp) << std::endl;
	std::cout << "weight:\n" << linear_layer->weight << std::endl;

	// Non-linear Activations
	std::cout << "\nPyTorch Non-linear Activations\n";
	auto sample_data = torch::autograd::Variable( torch::tensor({1,2,-1,-1}, torch::TensorOptions(torch::kLong)) );
	auto myRelu = torch::relu(sample_data);
	std::cout << "relu on sample_data:\n" << myRelu << std::endl;

	// Neural Network
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	MyFirstNetwork my_network = MyFirstNetwork( 3, 2, 1);
	my_network->to(device);
	auto dict = my_network->named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
			std::cout<<(*n).key()<<std::endl;
	}

	// Loss
	auto input = torch::autograd::Variable(torch::randn(5)); //.requires_grad();
	input = input.view({1, -1});
	auto target = torch::autograd::Variable(torch::randn(5));
	target = target.view({1, -1});
	auto loss = torch::nn::functional::cross_entropy(input, target);
	loss.backward();

	// Optimizer
	torch::optim::Adam optimizer(my_network->parameters(), torch::optim::AdamOptions(0.01));

	return 0;
}


