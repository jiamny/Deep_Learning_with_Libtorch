
#include <iostream>
#include <iomanip>
#include <torch/torch.h>

class NeuralNetImpl : public torch::nn::Module {
 public:
    NeuralNetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes);

    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

TORCH_MODULE(NeuralNet);


NeuralNetImpl::NeuralNetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes)
    : fc1(input_size, hidden_size), fc2(hidden_size, num_classes) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor NeuralNetImpl::forward(torch::Tensor x) {
    x = torch::nn::functional::relu(fc1->forward(x));
    return fc2->forward(x);
}

int main() {
    std::cout << "FeedForward Neural Network\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t input_size = 784;
    const int64_t hidden_size = 500;
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double learning_rate = 0.001;

    const std::string MNIST_data_path = "/media/stree/localssd/DL_data/mnist2/MNIST/raw/";

    // MNIST Dataset
    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Neural Network model
    NeuralNetImpl model(input_size, hidden_size, num_classes);
    model.to(device);

    // Optimizer
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model.forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model.eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model.forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

