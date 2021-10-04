/*
 * LR_reg_BostonHousing.cpp
 *
 */

#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <exception>
#include "csvloader.h"
#include <unistd.h>

// for slice/subset tensor
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

// Linear Regression Model
// Network for Linear Regression. Contains only one Dense Layer
// Usage: auto net = std::make_shared<Net>(1, 1) [Note: Since in_dim = 1, and out_dim = 1]
struct Net : torch::nn::Module {
	Net(int in_dim, int out_dim) {
		fc1 = register_module("fc1", torch::nn::Linear(in_dim, out_dim)); //500));
//		fc2 = register_module("fc2", torch::nn::Linear(500, 500));
//		fc3 = register_module("fc3", torch::nn::Linear(500, 200));
//		fc4 = register_module("fc4", torch::nn::Linear(200, out_dim));
	}

	torch::Tensor forward(torch::Tensor x) {
		x = fc1->forward(x);
//		x = fc2->forward(x);
//		x = fc3->forward(x);
//		x = fc4->forward(x);
		return x;
	}

	torch::nn::Linear fc1{nullptr}; //, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};

// ---- sample
std::vector<int> sample(int n, int min, int max) {
	std::vector<int> out;

	if( n < 0 )
	     throw std::runtime_error("negative sample size");
	if( max < min )
	     throw std::runtime_error("invalid range");
	if( n > (max - min + 1) )
	     throw std::runtime_error("sample size larger than range");

	while( n > 0 ) {
	   float r = std::rand()/(RAND_MAX+1.0);
	   if( r*(max-min+1) < n ) {
		   out.push_back(min);
	       --n;
	   }
	   ++min;
	}

    return(out);
}

// ---- shuffle
void FisherYatesShuffle(std::vector<int> &indices){

    for (int i = indices.size() - 1; i >= 0 ; i--) {
    	// generate a random number j such that 0 <= j <= i
    	int j = rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
}


int main(int argc, char** argv) {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	// Load CSV data
	std::ifstream file;
	std::string path = "./data/BostonHousing.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// ste file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);
/*
	std::pair<std::vector<float>, std::vector<float>> out = process_data(file);
	std::vector<float> train = out.first;
	std::vector<float> label = out.second;

	auto train_label = torch::from_blob(label.data(), {int(label.size()), 1});
	auto train_data = torch::from_blob(train.data(), {int(label.size()), int(train.size()/label.size())});

	std::cout << "size = " << train_data.sizes() << '\n';

*/
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
	//std::cout << '\n';

	// Process Data, load features and labels for LR
	std::vector<float> trainData;
	std::vector<float> trainLabel;
	std::vector<float> testData;
	std::vector<float> testLabel;


	process_split_data(file, train_idx, trainData, trainLabel, testData, testLabel);

	std::cout << "Train size = " << trainData.size() << "\n";
	std::cout << "Test size = " << testData.size() << "\n";


	// Data created
	// Convert vectors to a tensor
	// These fields should not be hardcoded (506, 1, 13)
	auto train_label = torch::from_blob(trainLabel.data(), {int(trainLabel.size()), 1});
	auto train_data = torch::from_blob(trainData.data(), {int(trainLabel.size()), int(trainData.size()/trainLabel.size())});
	std::cout << "szie = " << train_label.data().sizes() << "\n";
	std::cout << "szie = " << train_data.data().size(0) << "\n";

	// Create Network
	auto net = std::make_shared<Net>(int(train_data.sizes()[1]), 1);
	torch::optim::SGD optimizer(net->parameters(), 0.001);

	// Train
	std::size_t n_epochs = 10;
	for (std::size_t epoch = 1; epoch <= n_epochs; epoch++) {
		auto out = net->forward(train_data);
		optimizer.zero_grad();

		auto loss = torch::mse_loss(out, train_label);
		float loss_val = loss.item<float>();

		loss.backward();
		optimizer.step();

		std::cout << "Loss: " << loss_val << std::endl;
	}

	return(0);
}
