// Include libraries
#include "dataSet.h"
//Image data avaiable at: http://www.vision.caltech.edu/datasets/

int main(int argc, char** argv) {
	std::string datasetPath = "/media/stree/localssd/DL_data/Caltech_101/";
	std::string infoFilePath = "./data/Caltech_101_info.txt";

	int batch_size = 8;
//	auto data = readSplittedDataInfo(infoFilePath);

//	auto custom_dataset_init = CustomDataset(datasetPath, infoFilePath, "train", 64, 3);
//	custom_dataset_init.show_sample(5);
//	auto train_set = custom_dataset_init.map(torch::data::transforms::Stack<>());

	auto train_set = myDataset(datasetPath, infoFilePath, "train", 64, 3).map(torch::data::transforms::Stack<>());
	auto train_size = train_set.size().value();
	std::cout << "train_size = " << train_size << std::endl;

	if( train_size > 0 ) {
	    std::cout << "------------------ Load by info file randomsampler ----------------------\n";
	    auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), batch_size);

	    for(auto &batch: *mdataloader){
	        auto data = batch.data;
	        auto target = batch.target;
	        std::cout << data.sizes() << target << std::endl;
	    }
	}

	std:: cout << "Done!\n";
	return 0;
}


