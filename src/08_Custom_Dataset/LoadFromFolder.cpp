#include "dataSet.h"

// Download dataset from: https://download.pytorch.org/tutorial/hymenoptera_data.zip

int main(int argc, char *argv[])
{
    int batch_size = 8;
    std::string image_dir = "/media/stree/localssd/DL_data/hymenoptera_data/train";

    auto mdataset = myDataset(image_dir, ".jpg", 64, 3).map(torch::data::transforms::Stack<>());

    std::cout << "Data size = " << mdataset.size().value() << std::endl;

    std::cout << "------------------ RandomSampler ----------------------\n";
    auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(mdataset), batch_size);


    for(auto &batch: *mdataloader){
        auto data = batch.data;
        auto target = batch.target;
        std::cout << data.sizes() << target << std::endl;
    }

    auto mdatasetS = myDataset(image_dir, ".jpg", 64, 3).map(torch::data::transforms::Stack<>());
    std::cout << "Data size = " << mdatasetS.size().value() << std::endl;

    std::cout << "------------------ SequentialSampler ----------------------\n";
    auto mdataloaderS = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(mdatasetS), batch_size);

    for(auto &batch: *mdataloaderS){
        auto data = batch.data;
        auto target = batch.target;
        std::cout << data.sizes() << target << std::endl;
    }

    std::cout << "Done!\n";
    return 0;
}
