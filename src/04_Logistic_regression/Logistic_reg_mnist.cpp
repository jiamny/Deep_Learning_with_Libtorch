// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <matplot/matplot.h>
using namespace matplot;

using namespace torch::autograd;

int main() {
    std::cout << "Logistic Regression on fashion-MNIST data\n\n";

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
    const int64_t input_size = 784;
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 20;
    const double learning_rate = 0.001;

    const std::string MNIST_data_path = "/media/stree/localssd/DL_data/fashion_MNIST/";

    // MNIST Dataset (images and labels)
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

    // Logistic regression model
    torch::nn::Linear model(input_size, num_classes);

    model->to(device);

    // Loss and optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

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
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
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
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

    // ---- show some pred examples
	auto h = figure(true);
	h->width(h->width() * 2);
	h->height(h->height() * 2);
	h->x_position(5);
	h->y_position(5);

    auto dtype_option = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

    int rid = 0, cid = 0;
    for (auto& batch : *test_loader) {
    	auto idata = batch.data.clone();
    	auto data = batch.data.view({batch_size, -1}).to(device);;
    	auto target = batch.target.to(device);

    	std::cout << "idata.sizes() = " << idata.size(0) << "\n";
    	std::cout << "target.size() = " << target.size(0) << "\n";

    	std::cout << data.data().sizes() << "\n";

    	auto output = model->forward(data);
    	auto loss = torch::nn::functional::cross_entropy(output, target);
        auto prediction = output.argmax(1);

    	for( int j = 10; j < 22; j++ ) {

        	auto image = idata.data()[j].squeeze();	//view({-1,1}).to(dtype_option);
        	//std::cout << image.data().sizes() << "\n";
        	//std::cout << image.options() << "\n";

        	int type_id = target.cpu().data()[j].item<int64_t>();
        	int pred_type_id = prediction.cpu().data()[j].item<int64_t>();

        	int ncols = 28, nrows = 28;
        	std::vector<std::vector<double>> C;
        	for( int i = 0; i < nrows; i++ ) {
        		std::vector<double> c;
        		for( int j = 0; j < ncols; j++ )
        			c.push_back(image[i][j].item<double>());
        		C.push_back(c);
        	}

        	matplot::subplot(3, 4, j - 10);
        	matplot::title(fashionMap.at(type_id).c_str());
        	matplot::image(C);
        	matplot::axis(false);
        	h->draw();
    	}
    	matplot::show();

    	break;
    }

    std::cout << "Done!\n";
    return 0;
}
