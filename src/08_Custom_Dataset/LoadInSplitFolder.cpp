
#include "dataSet.h"

int main(int argc, char *argv[])
{
    int batch_size = 2;
    std::string image_dir = "./data/hymenoptera_data/train";

    // --- also get label to name map
    std::map<int, std::string> label_to_name;
    auto train_set = myDataset(image_dir, ".jpg", 64, 3, label_to_name).map(torch::data::transforms::Stack<>());
    auto train_size = train_set.size().value();
    std::cout << "Data size = " << train_size << std::endl;

    std::cout << "------------------ Train set RandomSampler ----------------------\n";
    auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), batch_size);

    for(auto &batch: *trainloader){
        auto data = batch.data;
        auto target = batch.target;
        std::cout << data.sizes() << target << std::endl;
    }

    std::cout << "------------------ Label to names ----------------------\n";
    for( int i = 0; i < label_to_name.size(); i++ ) {
       	std::cout << "key = " << i << " name = " << label_to_name.at(i) << std::endl;
    }

    std::cout << "Done!\n";
    return 0;
}
