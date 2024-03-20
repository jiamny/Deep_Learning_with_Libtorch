/*
 * Softmax_regression.cpp
 *
 */

#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <exception>
#include "../csvloader.h"
#include "../LRdataset.h"
#include <unistd.h>

#include <matplot/matplot.h>
using namespace matplot;

// ---- shuffle
void FisherYatesShuffle(std::vector<int> &indices){

    for (int i = indices.size() - 1; i >= 0 ; i--) {
    	// generate a random number j such that 0 <= j <= i
    	int j = rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
}

int main() {
    std::cout << "Logistic Regression iris\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    auto dtype_option = torch::TensorOptions().dtype(torch::kDouble).device(device);

	// Load CSV data
	std::ifstream file;
	std::string path = "./data/iris.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records += 1; 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::vector<int> indices;
	for( int i = 0; i < num_records; i++ ) indices.push_back(i);

	// suffle the indices
	FisherYatesShuffle(indices);

	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.8);
	for( int i = 0; i < train_size; i++ ) {
		//std::cout << indices.at(i) << " ";
		train_idx.insert(indices.at(i));
	}

    // Create an unordered_map to hold label names
    std::unordered_map<std::string, int> irisMap;
    irisMap.insert({"Iris-setosa", 0});
    irisMap.insert({"Iris-versicolor", 1});
    irisMap.insert({"Iris-virginica", 2});

    std::unordered_map<std::string, int> iMap={{"", -1}};
    std::cout << "irisMap[]: " << irisMap["Iris-setosa"] << " iMap.size(): " << iMap["Iris-versicolor"] << '\n';


	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> dt =
			process_split_data2(file, train_idx, irisMap, false, false, false);

	std::cout << "Train size = " << std::get<0>(dt).sizes() << "\n";
	std::cout << "Test size = " << std::get<2>(dt).sizes() << "\n";


	// Hyperparameters
    int num_epochs = 50;
    int batch_size = 8;
    int num_classes = 3;
    torch::Tensor train_dt = std::get<0>(dt).to(torch::kFloat), train_label = std::get<1>(dt).to(torch::kLong);
    torch::Tensor test_dt  = std::get<2>(dt).to(torch::kFloat), test_label  = std::get<3>(dt).to(torch::kLong);

    std::cout << "tr_L: " << train_label << '\n';

    torch::nn::Linear net(4, num_classes);
    net->to(device);

    if (auto M = dynamic_cast<torch::nn::LinearImpl*>(net.get())) {
    	   torch::nn::init::normal_(M->weight, 0, 0.01);
    }

	// loss
	torch::nn::CrossEntropyLoss criterion;

	// Optimization Algorithm
	auto trainer = torch::optim::SGD(net->parameters(), 0.1);

	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> test_loss;
	std::vector<double> test_acc;
	std::vector<double> xx;

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {
		net->train(true);
		torch::AutoGradMode enable_grad(true);

		 // Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
        int64_t num_train_samples = 0;

        auto dataloader =  data_iter(train_dt, train_label, batch_size);

        for(auto& batch : dataloader) {
			auto x = batch.first.to(device);;
			auto y = batch.second.flatten().to(device);

			auto y_hat = net->forward(x);

			auto loss = criterion(y_hat, y); //torch::cross_entropy_loss(y_hat, y);
			std::cout << "loss: " << loss.item<double>() << std::endl;

			// Update running loss
			epoch_loss += loss.item<double>() * x.size(0);

			// Calculate prediction
			auto prediction = y_hat.argmax(1);
			// Update number of correctly classified samples
			epoch_correct += (prediction == y).sum().item<int>();

			trainer.zero_grad();
			loss.backward();
			trainer.step();
			num_train_samples += x.size(0);
		}

		auto sample_mean_loss = (epoch_loss / num_train_samples);
		auto accuracy = static_cast<double>(epoch_correct *1.0 / num_train_samples);

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';

		train_loss.push_back((sample_mean_loss));
		train_acc.push_back(accuracy);

		std::cout << "Training finished!\n\n";
		std::cout << "Testing...\n";

		// Test the model
		net->eval();
		torch::NoGradGuard no_grad;

		double tst_loss = 0.0;
		epoch_correct = 0;
		int64_t num_test_samples = 0;

		auto test_loader =  data_iter(test_dt, test_label, batch_size);
		for(auto& batch : test_loader) {

			auto data = batch.first.to(device);
			auto target = batch.second.flatten().to(device);

			auto output = net->forward(data);

			auto loss = criterion(output, target); //torch::nn::functional::cross_entropy(output, target);
			tst_loss += loss.item<double>() * data.size(0);

			auto prediction = output.argmax(1);
			epoch_correct += prediction.eq(target).sum().item<int64_t>();

			num_test_samples += data.size(0);
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(epoch_correct * 1.0 / num_test_samples);
		auto test_sample_mean_loss = tst_loss / num_test_samples;

		test_loss.push_back((test_sample_mean_loss));
		test_acc.push_back(test_accuracy);

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::ylim(ax1, {0.3, 0.99});
	matplot::plot(ax1, xx, train_loss, "b")->line_width(2);
	matplot::plot(ax1, xx, test_loss, "m-:")->line_width(2);
	matplot::plot(ax1, xx, train_acc, "g--")->line_width(2);
	matplot::plot(ax1, xx, test_acc, "r-.")->line_width(2);
    matplot::hold(ax1, false);
    matplot::xlabel(ax1, "epoch");
    matplot::legend(ax1, {"Train loss", "Test loss", "Train acc", "Test acc"});
    matplot::show();

    return(0);
}


