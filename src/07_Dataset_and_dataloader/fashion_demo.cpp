#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "fashion.h"


int main() {
    std::cout << "Load fashion data set ...\n";

    auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Create an unordered_map to hold label names
    std::unordered_map<int, std::string> fashionMap;
    fashionMap.insert({0, "T-shirt/top"});
    fashionMap.insert({1, "Trouser"});
    fashionMap.insert({2, "Pullover"});
    fashionMap.insert({3, "Dress"});
    fashionMap.insert({4, "Coat"});
    fashionMap.insert({5, "Sandal"});
    fashionMap.insert({6, "Short"});
    fashionMap.insert({7, "Sneaker"});
    fashionMap.insert({8, "Bag"});
    fashionMap.insert({9, "Ankle boot"});


    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 8;
    const size_t num_epochs = 50;
    const double learning_rate = 0.001;

    const std::string FASHION_data_path = "/media/stree/localssd/DL_data/fashion_MNIST/";

    // MNIST custom dataset
    auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
//        	.map(ConstantPad(4))
//			.map(RandomHorizontalFlip())
//			.map(RandomCrop({28, 28}))
			.map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    std::cout << "Number of samples = " << num_train_samples << std::endl;
    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
         std::move(train_dataset), batch_size);

    for(auto &batch: *train_loader){
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            std::cout << data.sizes()   << " " << data.device()   << std::endl;
            std::cout << target.sizes() << " " << target.device() << std::endl;
    }

    std::cout << "Done!\n";
    return 0;
}



