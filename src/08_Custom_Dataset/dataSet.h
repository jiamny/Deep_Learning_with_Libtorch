#ifndef DATASET_H
#define DATASET_H
#endif // DATASET_H
#include <ATen/ATen.h>
#include <iostream>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <dirent.h>           //get files in directory
#include <sys/stat.h>
#include <cmath>
#include <map>

using Data = std::vector<std::pair<std::string, int>>;

// get all files in the folder
std::vector<std::string> GetFilesInDirectory(const std::string &directory);

// read data already splited
void readSplittedDataInfo(std::string dir, std::string infoFilePath, std::string type,
							std::vector<std::string> &list_images, std::vector<int> &list_labels);

// get from a single folder and separate train and test data sets by ratio
void load_data_from_folder_and_split(std::string file_root, float train_pct, std::vector<std::string> &list_images,
										std::vector<int> &list_labels, std::vector<std::string> &test_images,
										std::vector<int> &test_labels, std::map<int, std::string> &label_names);

// get data set from train or test or valid folder
void load_data_from_split_folder(std::string file_root, std::vector<std::string> &list_images,
									std::vector<int> &list_labels, std::map<int, std::string> &label_names);

void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels);

void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images,
							std::vector<int> &list_labels, std::map<int, std::string> &label_names);


class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

public:
	CustomDataset(int img_size, int channels, std::vector<std::string> &img_paths, std::vector<int> &img_labels) {
		image_size = img_size;
		channel_size = channels;
		image_paths = img_paths;
		labels = img_labels;
	}

	// Override get() function to return tensor at location index
	torch::data::Example<> get(size_t index) override{
		std::string image_path = image_paths.at(index);
		cv::Mat image = cv::imread(image_path.c_str());
		cv::resize(image, image, cv::Size(image_size, image_size));
		int label = labels.at(index);
		torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, channel_size }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
		torch::Tensor label_tensor = torch::full({ 1 }, label);
		return {img_tensor.clone(), label_tensor.clone()};
	}

	// Return the length of data
	torch::optional<size_t> size() const override {
		return image_paths.size();
	};

	// Visualizes sample at the given index
	void show_sample(int index) {
		cv::Mat sample_img = cv::imread(image_paths[index].c_str());
		cv::resize(sample_img, sample_img, cv::Size(image_size, image_size));
		cv::imshow("Image",sample_img);
		cv::waitKey(0);
	}

private:
	std::vector<std::string> image_paths;
	std::vector<int> labels;

	std::string dir_path = "";
	int image_size = 0;
	int channel_size = 0;
};


class myDataset:public torch::data::Dataset<myDataset>{
public:
    myDataset(std::string image_dir, std::string type, int img_size, int channels){
    	dir_path = image_dir;
    	image_size = img_size;
    	channel_size = channels;

        load_data_from_folder(image_dir, std::string(type), image_paths, labels);
        std::cout << image_size << " " << channel_size << std::endl;
    }

    myDataset(std::string dir, std::string type, int img_size, int channels, std::map<int, std::string> &label_names) {
        dir_path = dir;
        image_size = img_size;
        channel_size = channels;

        load_data_from_folder(dir, std::string(type), image_paths, labels, label_names);
    };

    myDataset(std::string dir, std::string info_path, std::string type, int img_size, int channels) {
       	dir_path = dir;
       	image_size = img_size;
       	channel_size = channels;
        readSplittedDataInfo(dir, info_path, std::string(type), image_paths, labels);
        std::cout << image_size << " " << channel_size << std::endl;
    }

    myDataset(int img_size, int channels, std::vector<std::string> &img_paths, std::vector<int> &img_labels) {
        image_size = img_size;
        channel_size = channels;
        image_paths = img_paths;
        labels = img_labels;
    };

    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override{
        std::string image_path = image_paths.at(index);
        cv::Mat image = cv::imread(image_path.c_str());
        cv::resize(image, image, cv::Size(image_size, image_size));
        int label = labels.at(index);
        torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, channel_size }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({ 1 }, label);
        return {img_tensor.clone(), label_tensor.clone()};
    }
    // Return the length of data
    torch::optional<size_t> size() const override {
        return image_paths.size();
    };

    // Visualizes sample at the given index
    void show_sample(int index) {
  	    cv::Mat sample_img = cv::imread(image_paths[index].c_str());
  	    cv::resize(sample_img, sample_img, cv::Size(image_size, image_size));
        cv::imshow("Image",sample_img);
        cv::waitKey(0);
    }

private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;

    std::string dir_path = "";
    int image_size = 0;
    int channel_size = 0;
};

