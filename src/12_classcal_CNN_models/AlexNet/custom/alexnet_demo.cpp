
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>

#include <iostream>
#include <stdexcept>
#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

#include "../../dataSet.h"
#include "alexnet.h"


int main() {
    const int64_t random_seed{0};
    const float learning_rate{1e-1};
    const float beta1 {0.5};
    const float beta2 {0.99};
    const int64_t num_epochs{5};
    torch::manual_seed(random_seed);
    // torch::cuda::manual_seed(random_seed);
    const int64_t batch_size{4};
    const int64_t num_workers{2};

    bool saveModel{false};

    std::string model_dir="./models";
    std::string model_file_name="./models/custom_alexnet.pt";
    std::string model_file_path = model_dir + "/" + model_file_name;

    const std::string train_data_path = "./data/hymenoptera_data/train";
    const std::string test_data_path = "./data/hymenoptera_data/val";
    std::vector<std::string> classes;

    std::cout << "Loading train dataset...\n";
    auto train_set = dataSetClc(train_data_path, ".png", 227, classes);

    std::string cls[classes.size()];
    std::copy(classes.begin(), classes.end(), cls);
    for(int i=0; i < classes.size(); i++) {
       std::cout << cls[i] << '\n';
    }

    std::cout << "Loading test dataset...\n";
    auto test_set = dataSetClc(test_data_path, ".png", 227, cls, classes.size());

    torch::Device device("cpu");
    if (torch::cuda::is_available()) {
        device = torch::Device("cuda:0");
    }

    // This might be different from the PyTorch API.
    // We did transform for the dataset directly instead of doing transform in
    // dataloader. Currently there is no augmentation options such as random
    // crop.
    auto train_set_transformed =
        train_set
//            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
		    .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
            .map(torch::data::transforms::Stack<>());

    auto test_set_transformed =
        test_set
//            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
		    .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
            .map(torch::data::transforms::Stack<>());
// std::unique_ptr<StatelessDataLoader<Dataset, Sampler>>>
    auto train_data_loader = torch::data::make_data_loader(
        std::move(train_set_transformed), torch::data::DataLoaderOptions()
                                              .batch_size(batch_size)
                                              .workers(num_workers)
                                              .enforce_ordering(true));

    auto test_data_loader = torch::data::make_data_loader(
        std::move(test_set_transformed), torch::data::DataLoaderOptions()
                                             .batch_size(batch_size)
                                             .workers(num_workers)
                                             .enforce_ordering(true));


    // (5) Define Network
    MC_AlexNet model(classes.size());
    model->to(device);

    std::cout << "Training Model..." << std::endl;
//    model = train_model(model, train_data_loader, test_data_loader, device,
//                        learning_rate, beta1, beta2, num_epochs);
//    model->train();
//    model->to(device);

    torch::nn::CrossEntropyLoss criterion{};

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(),
            		torch::optim::AdamOptions(learning_rate).betas({beta1, beta2}));

    //    torch::optim::SGD optimizer{model->parameters(),
    //                                torch::optim::SGDOptions(/*lr=*/learning_rate)
    //                                    .momentum(0.9)
    //                                    .weight_decay(1e-4)};

    // Requires LibTorch >= 1.90
    // torch::optim::LRScheduler scheduler{optimizer, 50, 0.1};

    // Train the network
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;

        int i = 0;
        for (auto& batch : *train_data_loader) {
            // get the inputs; data is a list of [inputs, labels]
            auto inputs = batch.data.to(device);
//            std::cout << inputs << std::endl;
            // what():  1D target tensor expected, multi-target not supported
            auto labels = batch.target.to(device).flatten(0, -1); // 1D target tensor expected
//            std::cout << labels << std::endl;

            // zero the parameter gradients
            optimizer.zero_grad();

            // forward + backward + optimize
            auto outputs = model->forward(inputs);
            auto loss = criterion(outputs, labels);
            loss.backward();
            optimizer.step();

            // print statistics
            running_loss += loss.item<double>();
            if (i % 20 == 19) {  // print every 2000 mini-batches
                std::cout << "[" << epoch + 1 << ", " << i + 1 << "] loss: "
                    << running_loss / 20 << '\n';
                running_loss = 0.0;
            }
            i++;
        }
    }
    std::cout << "Finished Training\n\n";

    if( saveModel ) {
    	torch::save(model, model_file_name);

    	// Test the network on the test data
    	model = MC_AlexNet(classes.size());
    	torch::load(model, model_file_name);
    }

    int correct = 0;
    int total = 0;
    for (const auto& batch : *test_data_loader) {
        auto images = batch.data.to(device);
        // what():  1D target tensor expected, multi-target not supported
        auto labels = batch.target.to(device).flatten(0, -1);

        auto outputs = model->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        total += labels.size(0);
 //       std::cout << "pred => " << predicted << std::endl;
 //       std::cout << "label => " << labels   << std::endl;
 //       std:: cout << (predicted == labels).sum().item<int>() << std::endl;

        correct += (predicted == labels).sum().item<int>();
    }

    std::cout << "Accuracy of the network on the 150 test images: "
        << (100 * correct / total) << "%\n\n";

    float class_correct[classes.size()];
    float class_total[classes.size()];
    for(int j =0; j < classes.size(); j++) {
    	class_correct[j] = 0.0;
    	class_total[j]   = 0.0;
    }

    torch::NoGradGuard no_grad;

    for (const auto& batch : *test_data_loader) {
        auto images = batch.data.to(device);
        auto labels = batch.target.to(device).flatten(0, -1);

        auto outputs = model->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
//        std::cout << "squeeze => " << (predicted == labels).squeeze() << std::endl;
        auto c = (predicted == labels).squeeze();

//        std::cout << "c.size => " << c.dim() << " labels => " << labels << std::endl;
        if( c.dim() > 0 ) {
        	for (int i = 0; i < c.dim(); ++i) {
        		//std::cout << "labels[i] -> " << labels[i].item<int>() << " c.itemsize -> " << c[i].itemsize() << std::endl;
        		if( c[i].itemsize() > 0 ) {
        			auto label = labels[i].item<int>();
        			class_correct[label] += c[i].item<float>();
        			class_total[label] += 1;
        		}
        	}
        }
    }

    for (int i = 0; i < classes.size(); ++i) {
        std::cout << "Accuracy of " << cls[i] << " "
            << 100 * class_correct[i] / class_total[i] << "%\n";
    }

    std::cout << "Done!\n";
    return 0;
}
