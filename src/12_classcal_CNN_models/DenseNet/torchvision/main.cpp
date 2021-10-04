// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "densenet.h"
#include "../../dataSet.h"

int main() {
	 const int64_t batch_size = 100;
     // Hyper parameters
     const int64_t bn_size{4};
     const int64_t num_init_features{64};
     const int64_t growth_rate{32};
     const std::vector<int64_t> block_config = {6, 12, 24, 16};
     double drop_rate{0};
     const double learning_rate = 0.001;
     const size_t learning_rate_decay_frequency = 8;  // number of epochs after which to decay the learning rate
     const double learning_rate_decay_factor = 1.0 / 3.0;
     const int64_t random_seed{0};
     const size_t num_epochs{2};
   	 torch::manual_seed(random_seed);
   	 // torch::cuda::manual_seed(random_seed);
   	 const int64_t num_workers{2};
   	 const size_t imgSize{96};

   	bool saveModel{false};

     // Device
     auto cuda_available = torch::cuda::is_available();
     torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
     std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

     // test model

     DenseNet121 net(10, growth_rate, block_config, num_init_features, bn_size, drop_rate );
     net->to(device);
     auto dict = net->named_parameters();
     for (auto n = dict.begin(); n != dict.end(); n++) {
    	 std::cout<<(*n).key()<<std::endl;
     }

     std::cout << "Test model ..." << std::endl;
     torch::Tensor x = torch::randn({1,3,96,96});
     torch::Tensor y = net(x);
     std::cout << y << std::endl;

   	 std::string model_dir="./models";
   	 std::string model_file_name="./models/torchvision_densenet.pt";
   	 std::string model_file_path = model_dir + "/" + model_file_name;

   	 const std::string train_data_path = "./data/CIFAR10/train";
   	 const std::string test_data_path = "./data/CIFAR10/test";
   	 std::vector<std::string> classes;

   	 std::cout << "Loading train dataset...\n";
   	 auto train_set = dataSetClc(train_data_path, ".png", imgSize, classes);
   	 size_t num_train_samples = train_set.size().value();

   	 std::string cls[classes.size()];
   	 std::copy(classes.begin(), classes.end(), cls);
   	 for(int i=0; i < classes.size(); i++) {
   	    std::cout << cls[i] << '\n';
   	 }

   	 std::cout << "Loading test dataset...\n";
   	 auto test_set = dataSetClc(test_data_path, ".png", imgSize, cls, classes.size());

   	 // This might be different from the PyTorch API.
   	 // We did transform for the dataset directly instead of doing transform in
   	 // dataloader. Currently there is no augmentation options such as random
   	 // crop.
   	 auto train_set_transformed =
   	        train_set
   	            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
   	//		    .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
   	            .map(torch::data::transforms::Stack<>());
   	 size_t num_test_samples= test_set.size().value();

   	 auto test_set_transformed =
   	        test_set
   	            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
   	//		    .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
   	            .map(torch::data::transforms::Stack<>());

   	// std::unique_ptr<StatelessDataLoader<Dataset, Sampler>>>
   	auto train_loader = torch::data::make_data_loader(
   	        std::move(train_set_transformed), torch::data::DataLoaderOptions()
   	                                              .batch_size(batch_size)
   	                                              .workers(num_workers)
   	                                              .enforce_ordering(true));

   	auto test_loader = torch::data::make_data_loader(
   	        std::move(test_set_transformed), torch::data::DataLoaderOptions()
   	                                             .batch_size(batch_size)
   	                                             .workers(num_workers)
   	                                             .enforce_ordering(true));
    // Model
    DenseNet121 model( classes.size(), growth_rate, block_config, num_init_features, bn_size, drop_rate );
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    auto current_learning_rate = learning_rate;

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            // Transfer images and target labels to device
            auto data = batch.data.to(device);
            auto target = batch.target.to(device).flatten(0, -1);

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

        // Decay learning rate
        if ((epoch + 1) % learning_rate_decay_frequency == 0) {
            current_learning_rate *= learning_rate_decay_factor;
            static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
                .options()).lr(current_learning_rate);
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";

    if( saveModel ) {
    	torch::save(model, model_file_name);

    	//Test the network on the test data
        model = DenseNet121( classes.size(), growth_rate, block_config, num_init_features, bn_size, drop_rate );
        torch::load(model, model_file_name);
    }

    std::cout << "Testing...\n";

    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device).flatten(0, -1);

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

    std::cout << "Done!\n";
    return 0;
}
