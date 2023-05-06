// Copyright 2020-present pytorch-cpp Authors
// Original: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "nnet.h"
#include "cifar10.h"

int main() {
    std::cout << "Deep Learning with PyTorch: A 60 Minute Blitz\n\n";
    std::cout << "Training a Classifier\n\n";

	// Device
	auto cuda_available = torch::cuda::is_available();

	torch::Device device = torch::Device(torch::kCPU);

	if( cuda_available ) {
		int gpu_id = 0;
		device = torch::Device(torch::kCUDA, gpu_id);

		if(gpu_id >= 0) {
			if(gpu_id >= torch::getNumGPUs()) {
				std::cout << "No GPU id " << gpu_id << " abailable, use CPU." << std::endl;
				device = torch::Device(torch::kCPU);
				cuda_available = false;
			} else {
				device = torch::Device(torch::kCUDA, gpu_id);
			}
		} else {
			device = torch::Device(torch::kCPU);
			cuda_available = false;
		}
	}


	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::cout << device << '\n';

	int batch_size = 4;

    // Loading and normalizing CIFAR10
    const std::string CIFAR_data_path = "/media/stree/localssd/DL_data/cifar/cifar10";

    auto train_dataset = CIFAR10(CIFAR_data_path)
        .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
        .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    std::string classes[10] = {"plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"};

    // Define a Convolutional Neural Network
    NetImpl net = NetImpl();
    net.to(device); //torch::kCPU);

    // // Define a Loss function and optimizer
    torch::nn::CrossEntropyLoss criterion;
    torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));

    // Train the network
    net.train();
   	torch::AutoGradMode enable_grad(true);

    for (size_t epoch = 0; epoch < 20; ++epoch) {
        double running_loss = 0.0;

        int i = 0;
        for (auto& batch : *train_loader) {
            // get the inputs; data is a list of [inputs, labels]
            auto inputs = batch.data.to(device);
            auto labels = batch.target.to(device);

            // zero the parameter gradients
            optimizer.zero_grad();

            // forward + backward + optimize
            auto outputs = net.forward(inputs);
            auto loss = criterion(outputs, labels);
            loss.backward();
            optimizer.step();

            // print statistics
            running_loss += loss.item<double>();
            if (i % 2000 == 1999) {  // print every 2000 mini-batches
                std::cout << "[" << epoch + 1 << ", " << i + 1 << "] loss: "
                    << running_loss / 2000 << '\n';
                running_loss = 0.0;
            }
            i++;
        }
    }
    std::cout << "Finished Training\n\n";
/*
    std::string PATH = "./cifar_net.pth";
    torch::save(net, PATH);

    // Test the network on the test data
    net = Net();
    torch::load(net, PATH);
*/
    net.eval();
    torch::NoGradGuard no_grad;

    int correct = 0;
    int total = 0;
    for (const auto& batch : *test_loader) {
        auto images = batch.data.to(device);
        auto labels = batch.target.to(device);

        auto outputs = net.forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        total += labels.size(0);
        correct += (predicted == labels).sum().item<int>();
    }

    std::cout << "Accuracy of the network on the 10000 test images: "
        << (100 * correct / total) << "%\n\n";

    int class_sz = sizeof(classes) / sizeof(classes[0]);
    std::cout << class_sz << '\n';

    int class_correct[class_sz];
    int class_total[class_sz];
    for(int n = 0; n < class_sz; n++) {
    	class_correct[n] = 0;
    	class_total[n] = 0;
    }

    for (const auto& batch : *test_loader) {
        auto images = batch.data.to(device);
        auto labels = batch.target.to(device);

        auto outputs = net.forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        auto c = (predicted == labels).squeeze();


        for (int i = 0; i < images.size(0); ++i) {
            auto label = labels[i].item<int>();
            class_correct[label] += c[i].item<int>();
            class_total[label] += 1;
        }

    }

    for (int i = 0; i < class_sz; ++i) {
        std::cout << "Accuracy of " << classes[i] << " "
            << 100.0 * class_correct[i] / class_total[i] << "%\n";
    }
    std::cout << "Done!\n";
}
