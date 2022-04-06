#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include "../../csvloader.h"
#include "../../LRdataset.h"
#include "../../matplotlibcpp.h"
namespace plt = matplotlibcpp;

std::pair<std::vector<float>, std::vector<float>> getFeaturesAndLabels(std::string path) {
	std::ifstream file;
	file.open(path, std::ios_base::in);

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// ste file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	return( process_data(file) );
}

torch::nn::Sequential create_net() {
	torch::nn::Sequential net ({
	        {"linear1", torch::nn::Linear(15,20)},
	        {"linear2", torch::nn::Linear(20,15)},
	        {"relu2", torch::nn::ReLU()},
			{"linear3", torch::nn::Linear(15,1)},
			{"sigmoid", torch::nn::Sigmoid()}
	    });
    return net;
}

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	torch::manual_seed(1000);

	// Load train CSV data
	std::ifstream file;
	std::string path = "./data/titanic/preprocessed_train.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}
	file.close();

	std::vector<float> trainData;
	std::vector<float> trainLabel;
	std::vector<float> testData;
	std::vector<float> testLabel;

	std::pair<std::vector<float>, std::vector<float>> train_datas = getFeaturesAndLabels(path);
	trainData  = train_datas.first;
	trainLabel = train_datas.second;
	std::cout << trainData.size() << std::endl;

	// Convert vectors to a tensor
	auto train_label = torch::from_blob(trainLabel.data(), {int(trainLabel.size()), 1}).clone();
	auto train_data = torch::from_blob(trainData.data(), {int(trainLabel.size()), int(trainData.size()/trainLabel.size())}).clone();
	std::cout << "size = " << train_label.data().sizes() << "\n";
	std::cout << "size = " << train_data.data().size(0) << "\n";
	std::cout << "train_data[0,:]\n" << train_data[0] << "\n";

	path = "./data/titanic/preprocessed_test.csv";
	file.open(path, std::ios_base::in);
	// Exit if file not opened successfully
	if (!file.is_open()) {
			std::cout << "File not read successfully" << std::endl;
			std::cout << "Path given: " << path << std::endl;
			return -1;
	}
	file.close();

	std::pair<std::vector<float>, std::vector<float>> test_datas = getFeaturesAndLabels(path);
	testData  = test_datas.first;
	testLabel = test_datas.second;
	std::cout << testData.size() << std::endl;

	// Convert vectors to a tensor
	auto test_label = torch::from_blob(testLabel.data(), {int(testLabel.size()), 1}).clone();
	auto test_data = torch::from_blob(testData.data(), {int(testLabel.size()), int(testData.size()/testLabel.size())}).clone();
	std::cout << "size = " << test_label.data().sizes() << "\n";
	std::cout << "size = " << test_data.sizes() << "\n";
	std::cout << "test_data[0,:]\n" << test_data[0] << "\n";

	size_t batch_size = 8;

	auto train_dataset = LRdataset(train_data, train_label)
					   .map(torch::data::transforms::Stack<>());

	auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		                   std::move(train_dataset), batch_size);

	for( auto& batch : *train_data_loader ) {
		auto X = batch.data;
		auto y = batch.target;
		std::cout << "X = \n" << X << "\n";
		std::cout << "y = \n" << y << "\n";
	    break;
	}

	auto test_dataset = LRdataset(test_data, test_label)
						   .map(torch::data::transforms::Stack<>());

	auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			                   std::move(test_dataset), batch_size);

	// ---- NN model
	auto net = create_net();

	std::cout << net << std::endl;

	auto loss_func = torch::nn::BCELoss();
	auto optimizer = torch::optim::Adam(net->parameters(), 0.01);

	size_t epochs = 10;
	size_t log_step_freq = 30;
	float loss_sum;
	int step;

	std::vector<float> v_epoch, v_train_loss, v_val_loss;

	for( size_t epoch = 0;  epoch < epochs; epoch++ ) {
		// train -------------------------------------------------
		net->train();
		loss_sum = 0.0;
		step = 0;

		for( auto& batch : *train_data_loader ) {
			auto features = batch.data;
			auto labels = batch.target;

			optimizer.zero_grad();

			auto predictions = net->forward(features);
			auto loss = loss_func(predictions,labels);

			loss.backward();
			optimizer.step();

			loss_sum += loss.item<float>();

			if( (step + 1) % log_step_freq == 0)
				std::cout <<"step = " << (step + 1) << " loss: " << loss_sum/(step + 1) << std::endl;
			step++;
		}

		// validation ---------------------------------------------
		net->eval();
		float val_loss_sum = 0.0;
		int val_step = 0;

		for( auto& batch : *test_data_loader ) {
			auto features = batch.data;
			auto labels = batch.target;

			torch::NoNamesGuard no_grad();
			auto predictions = net->forward(features);
			auto val_loss = loss_func(predictions,labels);
			val_loss_sum += val_loss.item<float>();
			val_step++;
		}
		std::cout << "\nEPOCH = " << epoch << ", loss = " << loss_sum/step << ", val_loss = " << val_loss_sum/val_step << "\n";
		v_epoch.push_back(epoch*1.0);
		v_train_loss.push_back(loss_sum/step);
		v_val_loss.push_back(val_loss_sum/val_step);
	}
	std::cout << "Finished Training...\n";

	plt::figure_size(800, 600);
	plt::named_plot("train loss", v_epoch, v_train_loss, "b");
	plt::named_plot("valid loss", v_epoch, v_val_loss, "r-.");
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}




