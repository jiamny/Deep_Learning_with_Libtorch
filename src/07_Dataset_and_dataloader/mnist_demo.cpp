#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "mnist.h"

#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main() {
    std::cout << "Load fashion data set ...\n";

    auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 8;
    const size_t num_epochs = 50;
    const double learning_rate = 0.001;

    const std::string mnist_data_path = "./data/MNIST/";

    // MNIST custom dataset
    auto train_dataset = MNIST(mnist_data_path, MNIST::Mode::kTrain)
        	.map(ConstantPad(4))
			.map(RandomHorizontalFlip())
			.map(RandomCrop({28, 28}))
			.map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    std::cout << "Number of samples = " << num_train_samples << std::endl;
    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
         std::move(train_dataset), batch_size);

    for(auto &batch: *train_loader){
            auto data = batch.data;
            auto target = batch.target;
            std::cout << data.sizes() << " " << target << std::endl;
    }

    std::cout << "Done!\n";
    return 0;
}



