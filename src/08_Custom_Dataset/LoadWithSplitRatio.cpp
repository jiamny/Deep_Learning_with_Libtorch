// Include libraries
#include "dataSet.h"


int main(int argc, char** argv) {
	std::string datasetPath = "./data/Caltech_101/";
	std::string infoFilePath = "./data/Caltech_101_info.txt";

	int batch_size = 8;

    // Dataset load and split to train and test
	float train_percent = 0.8;
	std::map<int, std::string> label_to_name;
	std::vector<std::string> train_imgpaths;
	std::vector<std::string> test_imgpaths;
	std::vector<int> train_labels;
	std::vector<int> test_labels;

	load_data_from_folder_and_split(datasetPath, train_percent, train_imgpaths, train_labels,
			test_imgpaths, test_labels, label_to_name);

	// Use the map
	std::map<int, std::string>::const_iterator iter = label_to_name.find(102);
	if (iter != label_to_name.end()){
	    // iter->second contains your string
	    std::cout << iter->first  << " " << iter->second << std::endl;
	}
/*
	//auto train_set = custom_dataset_init.map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
	auto train_set_init = CustomDataset(64, 3, train_imgpaths, train_labels);
	train_set_init.show_sample(10);
	auto train_set = train_set_init.map(torch::data::transforms::Stack<>());
*/
	auto train_set_init = myDataset(64, 3, train_imgpaths, train_labels);
	train_set_init.show_sample(10);
	auto train_set = train_set_init.map(torch::data::transforms::Stack<>());


//	auto train_set = myDataset(64, 3, train_imgpaths, train_labels).map(torch::data::transforms::Stack<>());
	std::cout << "train_size = " << train_set.size().value() << std::endl;

	auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), batch_size);

	for(auto &batch: *mdataloader){
	   auto data = batch.data;
	   auto target = batch.target;
	   std::cout << data.sizes() << target << std::endl;
	}

	std:: cout << "Done!\n";
	return 0;
}


