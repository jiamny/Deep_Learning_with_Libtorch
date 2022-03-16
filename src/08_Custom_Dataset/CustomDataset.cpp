// Include libraries
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <string>

/* Convert and Load image to tensor from location argument */
torch::Tensor read_data(std::string loc) {
  // Read Data here
  // Return tensor form of the image
  cv::Mat img = cv::imread(loc, 1);
	cv::resize(img, img, cv::Size(1920, 1080), cv::INTER_CUBIC);
	std::cout << "Sizes: " << img.size() << std::endl;
	torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
	img_tensor = img_tensor.permute({2, 0, 1}); // Channels x Height x Width

	return img_tensor.clone();
}

/* Converts label to tensor type in the integer argument */
torch::Tensor read_label(int label) {
    // Read label here
    // Convert to tensor and return
    torch::Tensor label_tensor = torch::full({1}, label);
	return label_tensor.clone();
}

/* Loads images to tensor type in the string argument */
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
    std::cout << "Reading Images..." << std::endl;
    // Return vector of Tensor form of all the images
    std::vector<torch::Tensor> states;
	for (std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
		torch::Tensor img = read_data(*it);
		states.push_back(img);
	}
	return states;
}

/* Loads labels to tensor type in the string argument */
std::vector<torch::Tensor> process_labels(std::vector<std::string> list_labels) {
	std::cout << "Reading Labels..." << std::endl;
    // Return vector of Tensor form of all the labels
	std::vector<torch::Tensor> labels;
	for (std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
		torch::Tensor label = read_label(*it);
		labels.push_back(label);
	}
	return labels;
}

class CustomDataset : public torch::data::dataset<CustomDataset> {
private:
  // Declare 2 vectors of tensors for images and labels
	std::vector<torch::Tensor> images, labels;
public:
  // Constructor
  CustomDataset(std::vector<std::string> list_images, std::vector<std::string> list_labels) {
    images = process_images(list_images);
    labels = process_labels(list_labels);
  };

  // Override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override {
    torch::Tensor sample_img = images.at(index);
    torch::Tensor sample_label = labels.at(index);
    return {sample_img.clone(), sample_label.clone()};
  };

  // Return the length of data
  torch::optional<size_t> size() const override {
    return labels.size();
  };
};

int main(int argc, char** argv) {
	std::vector<std::string> list_images; // list of path of images
	std::vector<int> list_labels; 		  // list of integer labels

  // Dataset init and apply transforms - None!
  auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());
}
