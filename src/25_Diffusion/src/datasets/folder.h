#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <fstream>

void load_imgs_from_folder(std::string folder, std::string image_type, std::vector<std::string> &list_images);

class ImageFolderDataset : public torch::data::Dataset<ImageFolderDataset, torch::data::TensorExample> {
private:
    std::vector<std::string> img_paths;
    std::tuple<int, int> img_dim; // img_height, img_width
public:
    ImageFolderDataset(std::string img_dir, std::tuple<int, int> img_dim, std::string img_type = "jpg") {
        load_imgs_from_folder(img_dir, img_type, img_paths);
        this->img_dim = img_dim;
    }

    torch::data::TensorExample get(size_t index) override {
        std::string image_path = img_paths.at(index);
        cv::Mat img = cv::imread(image_path);
        int inp_width;
        int inp_height;
        std::tie(inp_height, inp_width) = img_dim;
        cv::resize(img, img, {inp_width, inp_height});
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F, 2.0 / 255, -1.); // normalize to (-1, 1)
        auto img_tensor = torch::from_blob(img_float.data, {img.rows, img.cols, 3}, torch::kF32).permute({2, 0, 1});

        // In c++ function block, the memory assigned for variables will be lazy change.
        // That means two or more calling, the address of varibales in function will not be change, except the value.
        // So if the returned object whose member is a pointer created in function, it will pointer to the same address.
        // Which cause the 2nd calling overide 1st value.
        // Unfortunately, this is how Tensor.permute operates, it needs be clone to avoid this situation.
        // Because we don't want the two calling return same image.
        return {img_tensor.contiguous()};
    }


    torch::optional<size_t> size() const override {
        return img_paths.size();
    }
};
