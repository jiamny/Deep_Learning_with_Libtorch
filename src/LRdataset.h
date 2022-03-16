
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/utils.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <iostream>

class LRdataset : public torch::data::datasets::Dataset<LRdataset> {
 public:

    explicit LRdataset(std::pair<torch::Tensor, torch::Tensor> data_and_labels) {
    	features_ = std::move(data_and_labels.first);
    	labels_ = std::move(data_and_labels.second);
    }

    explicit LRdataset(torch::Tensor data, torch::Tensor labels) {
        features_ = data;
        labels_ = labels;
    }

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    // Returns all features.
    const torch::Tensor& features() const;

    // Returns all targets
    const torch::Tensor& labels() const;

 private:
    torch::Tensor features_;
    torch::Tensor labels_;
};

// 构建数据管道迭代器
std::list<std::pair<torch::Tensor, torch::Tensor>> data_iter(torch::Tensor X, torch::Tensor Y, int64_t batch_size=8);

