// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {

	std::cout << "Linear Regression\n\n";

	/*
	 * Simple C++ custom autograd function code throws error "CUDA error: driver shutting down"
	 * terminate called after throwing an instance of 'c10::Error'
	 *  what():  CUDA error: driver shutting down
	 *  CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
	 */

	// Device
	auto cuda_available = torch::cuda::is_available(); // add this line will let everything OK.
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';


    // Hyper parameters
    const int64_t input_size = 1;
    const int64_t output_size = 1;
    const size_t num_epochs = 200;
    const double learning_rate = 0.001;

    // Sample dataset
    auto x_train = torch::randint(0, 10, {15, 1}).to(torch::kFloat32).to(device);
    auto y_train = torch::randint(0, 10, {15, 1}).to(torch::kFloat32).to(device);

    // Linear regression model
    torch::nn::Linear model(input_size, output_size);
    model->to(device);
    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Forward pass
        auto output = model(x_train);

        auto loss = torch::nn::functional::mse_loss(output, y_train);

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs <<
                "], Loss: " << loss.item<float>() << "\n";
        }
    }

    std::cout << "Training finished!\n";

    return 0;
}
