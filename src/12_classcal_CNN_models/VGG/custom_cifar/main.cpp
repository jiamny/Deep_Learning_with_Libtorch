// Copyright 2020-present pytorch-cpp Authors
// Original: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "vgg.h"
#include "../../dataSet.h"

int main() {
    std::cout << "Training LeNet Classifier\n\n";
/*
    std::string classes[10] = {"plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"};
*/
    // Loading and normalizing CIFAR10
    const std::string train_data_path = "./data/CIFAR10/train";
    const std::string test_data_path = "./data/CIFAR10/test";
    std::vector<std::string> classes;
    const int64_t batch_size{4};

    bool saveModel{false};

    std::cout << "Loading train dataset...\n";
    auto train_dataset = dataSetClc(train_data_path, ".png", 224, classes)
		.map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                {0.2023, 0.1994, 0.2010}))
		.map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);

    std::string cls[classes.size()];
    std::copy(classes.begin(), classes.end(), cls);
    for(int i=0; i < classes.size(); i++) {
       std::cout << cls[i] << '\n';
    }

    std::cout << "Loading test dataset...\n";
    auto test_dataset = dataSetClc(test_data_path, ".png", 224, cls, classes.size())
		.map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                {0.2023, 0.1994, 0.2010}))
		.map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Define a Convolutional Neural Network
    size_t n_layers = 11;
    size_t nc = 3;
    size_t class_num = 10;
    bool  BN = false;
    MC_VGGNet net = MC_VGGNet(n_layers, nc, class_num, BN);
    net->to(torch::kCPU);

//     print(model)

//    auto input = torch::randn({8,3,32,32});
//    auto out = net(input);
//    std::cout << out.data() << std::endl;

    // // Define a Loss function and optimizer
    torch::nn::CrossEntropyLoss criterion;
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));

    // Train the network
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        double running_loss = 0.0;

        int i = 0;
        for (auto& batch : *train_loader) {
            // get the inputs; data is a list of [inputs, labels]
            auto inputs = batch.data.to(torch::kCPU);
            // what():  1D target tensor expected, multi-target not supported
            auto labels = batch.target.to(torch::kCPU).flatten(0, -1); // 1D target tensor expected
            //std::cout << labels << std::endl;

            // zero the parameter gradients
            optimizer.zero_grad();

            // forward + backward + optimize
            auto outputs = net->forward(inputs);
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

    if( saveModel ) {
    	std::string PATH = "./models/custom_cifar_net.pth";
    	torch::save(net, PATH);

    	// Test the network on the test data
    	net = MC_VGGNet(n_layers, nc, class_num, BN);
    	torch::load(net, PATH);
    }

    int correct = 0;
    int total = 0;
    for (const auto& batch : *test_loader) {
        auto images = batch.data.to(torch::kCPU);
        // what():  1D target tensor expected, multi-target not supported
        auto labels = batch.target.to(torch::kCPU).flatten(0, -1);

        auto outputs = net->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        total += labels.size(0);
 //       std::cout << "pred => " << predicted << std::endl;
 //       std::cout << "label => " << labels   << std::endl;
 //       std:: cout << (predicted == labels).sum().item<int>() << std::endl;

        correct += (predicted == labels).sum().item<int>();
    }

    std::cout << "Accuracy of the network on the 10000 test images: "
        << (100 * correct / total) << "%\n\n";

    float class_correct[classes.size()];
    float class_total[classes.size()];
    for(int j =0; j < classes.size(); j++) {
    	class_correct[j] = 0.0;
    	class_total[j]   = 0.0;
    }

    torch::NoGradGuard no_grad;

    for (const auto& batch : *test_loader) {
        auto images = batch.data.to(torch::kCPU);
        auto labels = batch.target.to(torch::kCPU).flatten(0, -1);

        auto outputs = net->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
 //       std::cout << "squeeze => " << (predicted == labels).squeeze() << std::endl;
        auto c = (predicted == labels).squeeze();

        if( c.dim() > 0 ) {
        	for (int i = 0; i < c.dim(); ++i) {
        		auto label = labels[i].item<int>();
        		class_correct[label] += c[i].item<float>();
        		class_total[label] += 1;
        	}
        }
    }

    for (int i = 0; i < classes.size(); ++i) {
        std::cout << "Accuracy of " << classes.at(i) << " "
            << 100 * class_correct[i] / class_total[i] << "%\n";
    }

    std::cout << "Done!\n";
    return 0;
}
