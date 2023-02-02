#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "fashion.h"
#include "transform.h"

#include "../matplotlibcpp.h"

using namespace torch::autograd;
namespace plt = matplotlibcpp;

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main() {
    std::cout << "Linear Regression FASHION dataset\n\n";

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

    auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(device);

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


    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 50;
    const double learning_rate = 0.001;

    const std::string FASHION_data_path = "/media/stree/localssd/DL_data/fashion_MNIST/";

    // MNIST custom dataset
    auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
        	.map(ConstantPad(4))
			.map(RandomHorizontalFlip())
			.map(RandomCrop({28, 28}))
			.map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
            .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
         std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
         std::move(test_dataset), batch_size);

    // Check image data
    for (auto& batch : *train_loader) {
    	auto data = batch.data.to(device);
    	auto target = batch.target.to(device);

    	std::cout << "data.size() = " << data.size(0) << "\n";
    	std::cout << "target.size() = " << target.size(0) << "\n";

    	std::cout << data.data().sizes() << "\n";

    	auto image = data.data()[12].view({-1,1}).to(dtype_option);
    	std::cout << image.data().sizes() << "\n";

    	int type_id = target.data()[12].item<int>();
    	std::cout << "type_id = " << type_id << " name = " << fashionMap.at(type_id) << "\n";

    	int ncols = 28, nrows = 28;
    	std::vector<float> z(image.data_ptr<float>(), image.data_ptr<float>() + image.numel());;
    	const float* zptr = &(z[0]);
    	const int colors = 1;
    	plt::imshow(zptr, nrows, ncols, colors);
    	plt::title(fashionMap.at(type_id));
    	plt::show();
    	break;
    }

    // Linear regression model
    torch::nn::Linear fc(28 * 28 , num_classes);
    fc->to(device);

    //torch::nn::CrossEntropyLoss  criterion;
    // Optimizer
    torch::optim::Adam optimizer(fc->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
    	// Initialize running metrics
    	 double running_loss = 0.0;
    	 size_t num_correct = 0;

    	 int idx = 0;
    	 for (auto& batch : *train_loader) {

    		 // Transfer images and target labels to device
    		 auto data = batch.data.to(device);
    		 auto target = batch.target.to(device);

    		 auto x = data.data().view({-1, 28*28}).to(dtype_option);

    		 optimizer.zero_grad();
    		 // Forward pass
    		 //auto output = fc->forward(data);
    		 auto output = fc(x);

    		 // Calculate loss
    		 auto loss = torch::nn::functional::cross_entropy(output, target);
    		 //auto loss = criterion(output, target);

    		 // Update running loss
    		 running_loss += loss.item<double>() * data.size(0);

    		 // Calculate prediction
    		 auto prediction = output.argmax(1);

             // Update number of correctly classified samples
             num_correct += prediction.eq(target).sum().item<int64_t>();

             // Backward pass and optimize
    		 loss.backward();
    		 optimizer.step();

    		 idx++;
    		 if( idx % 100 == 0 )
    			 std::cout << "Epach [" << (epoch+1) << "/" << num_epochs << "] "
				 << idx << " batch, loss = " << loss.item<double>() * data.size(0) << "\n";
    	 }
    	 auto sample_mean_loss = running_loss / num_train_samples;
    	 auto accuracy = static_cast<double>(num_correct) / num_train_samples;

    	 std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
    	            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    fc->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto x = data.data().view({-1, 28*28}).to(dtype_option);
        //auto output = fc->forward(data);
        auto output = fc(x);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
    return(0);
}




