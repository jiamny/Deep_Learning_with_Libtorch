#ifndef DATASET_H
#define DATASET_H
#endif // DATASET_H

#include <torch/script.h>
#include <torch/torch.h>
#include<vector>
#include <string>
//#include <io.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>           //get files in directory
#include <sys/stat.h>
#include <cmath>
#include <map>

// get all files in the folder
std::vector<std::string> GetFilesInDirectory(const std::string &directory);

//遍历该目录下的.jpg图片
void load_data_from_folder(std::string image_dir, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int class_label);
void load_data_and_classes_from_folder(std::string image_dir, std::string type, std::vector<std::string> &list_images,
							std::vector<int> &list_labels, std::vector<std::string> &class_names, int class_label);

void load_data_by_classes_from_folder(std::string image_dir, std::string type, std::vector<std::string> &list_images,
							std::vector<int> &list_labels, std::string names[], int num_class);

class dataSetClc:public torch::data::Dataset<dataSetClc>{
public:
 //   int num_classes = 0;
    dataSetClc(std::string image_dir, std::string type, int img_size){
    	image_size = img_size;
        load_data_from_folder(image_dir, std::string(type), image_paths, labels, 0); //, num_classes-1);
    }

    dataSetClc(std::string image_dir, std::string type, int img_size, std::vector<std::string> &class_names){
       	image_size = img_size;
        load_data_and_classes_from_folder(image_dir, std::string(type), image_paths, labels, class_names, 0);
    }

    dataSetClc(std::string image_dir, std::string type, int img_size, std::string names[], int num_classes){
           	image_size = img_size;
           	load_data_by_classes_from_folder(image_dir, std::string(type), image_paths, labels, names, num_classes);
    }
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override{
        std::string image_path = image_paths.at(index);
        cv::Mat image = cv::imread(image_path);
        cv::resize(image, image, cv::Size(image_size, image_size));
        int label = labels.at(index);
        torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, image.channels() }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({1}, label);
        return {img_tensor.clone().to(torch::kFloat32).div_(255), label_tensor.clone().to(torch::kInt64)};
    }
    // Return the length of data
    torch::optional<size_t> size() const override {
        return image_paths.size();
    };
private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
    int image_size;
};
