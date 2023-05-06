/*
 * Logistic_regression.cpp
 *
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Logistic Regression iris\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    auto dtype_option = torch::TensorOptions().dtype(torch::kDouble).device(device);

    // Create an unordered_map to hold label names
    std::unordered_map<std::string, int> irisMap;
    irisMap.insert({"Iris-setosa", 0});
    irisMap.insert({"Iris-versicolor", 1});
    irisMap.insert({"Iris-virginica", 2});

	// Hyperparameters
    size_t random_seed = 0;
    double learning_rate = 0.05;
    int num_epochs = 10;
    int batch_size = 8;

	// Architecture
    int num_features = 2;
    int num_classes = 3;

//    torch::data::DataSet

    return(0);
}


