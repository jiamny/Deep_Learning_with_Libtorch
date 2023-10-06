//
//
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "efficientnet.h"
#include "../../../07_Dataset_and_dataloader/cifar10.h"

int main() {

	std::cout << "EfficientNet\n\n";

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device = cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	EfficientNet net = EfficientNetB0(10, device);
	net->to(device);
	auto dict = net->named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
		std::cout<<(*n).key()<<std::endl;
		//std::cout<<(*n).value() <<std::endl;
	}

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1,3,32,32}).to(device);
	std::cout << "x: " << x.options() << '\n';
	torch::Tensor y = net->forward(x);
	std::cout << "y: " << y << std::endl;

	// Hyper parameters
	const int64_t image_size{32};
	const int64_t num_classes = 10;
	const int64_t batch_size = 100;
	const size_t num_epochs = 2;
	const double learning_rate = 0.001;
	const size_t learning_rate_decay_frequency = 8;  // number of epochs after which to decay the learning rate
	const double learning_rate_decay_factor = 1.0 / 3.0;

	bool saveBestModel{false};

	const std::string CIFAR_data_path = "/media/hhj/localssd/DL_data/cifar/cifar10/";
    std::string classes[10] = {"plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"};

	// CIFAR10 custom dataset
	auto train_dataset = CIFAR10(CIFAR_data_path)
			.map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
	        .map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
		    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
	        .map(torch::data::transforms::Stack<>());

	// Number of samples in the testset
	auto num_test_samples = test_dataset.size().value();
	std::cout << "num_test_samples: " << num_test_samples << std::endl;

	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	        std::move(train_dataset), batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	        std::move(test_dataset), batch_size);

	// Model
	EfficientNet model = EfficientNetB0(10, device);
	model->to(device);

	// Optimizer
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

	// Set floating point output precision
	std::cout << std::fixed << std::setprecision(4);

	auto current_learning_rate = learning_rate;
	double best_acc{0.0};
	std::string PATH = "./models/efficientnet.pth";

	// Train the model
	for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
		std::cout << "\nTraining...\n";

	    // Initialize running metrics
	    double running_loss = 0.0;
	    size_t num_correct = 0;

	    model->train();
	    torch::AutoGradMode enable_grad(true);

	    for (auto& batch : *train_loader) {
	        // Transfer images and target labels to device
	        auto data = batch.data.to(device);
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


	    std::cout << "Training finished!\n\n";
	    std::cout << "Testing...\n";

	    // Test the model
	    model->eval();
	    torch::NoGradGuard no_grad;

	    double test_loss = 0.0;
	    num_correct = 0;

	    for (const auto& batch : *test_loader) {
	        auto data = batch.data.to(device);
	        auto target = batch.target.to(device);

	        auto output = model->forward(data);

	        auto loss = torch::nn::functional::cross_entropy(output, target);
	        test_loss += loss.item<double>() * data.size(0);

	        auto prediction = output.argmax(1);
	        num_correct += prediction.eq(target).sum().item<int64_t>();
	    }

	    std::cout << "Testing finished!\n";

	    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
	    auto test_sample_mean_loss = running_loss / num_test_samples;

	    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

	    if( saveBestModel )
	    	if( test_accuracy > best_acc ) {
	    		torch::save(model, PATH);
	    		best_acc = test_accuracy;
	    	}
	}

	if( saveBestModel ) {
		model = EfficientNetB0(10, device);
		torch::load(model, PATH);
	}

    float class_correct[10];
    float class_total[10];
    for (int i = 0; i < 10; ++i) {
    	class_correct[i] = 0.0;
    	class_total[i] = 0.0;
    }

    torch::NoGradGuard no_grad;

    for (const auto& batch : *test_loader) {
        auto images = batch.data.to(device);
        auto labels = batch.target.to(device);

        auto outputs = model->forward(images);
        auto prediction = outputs.argmax(1);

        for (int i = 0; i < images.sizes()[0]; ++i) {
            auto label = labels[i].item<long>();
            if( label == prediction[i].item<long>() )
            	class_correct[label] += 1;
            class_total[label] += 1;
        }
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << "Accuracy of " << classes[i] << " "
                << 100 * class_correct[i] / class_total[i] << "%\n";
    }

    std::cout << "Done!\n";
    return 0;
}
