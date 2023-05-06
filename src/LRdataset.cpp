#include "LRdataset.h"


torch::data::Example<> LRdataset::get(size_t index) {
    return {features_[index], labels_[index]};
}

torch::optional<size_t> LRdataset::size() const {
    return features_.size(0);
}

const torch::Tensor& LRdataset::features() const {
    return features_;
}

const torch::Tensor& LRdataset::labels() const {
    return labels_;
}

// 构建数据管道迭代器
std::list<std::pair<torch::Tensor, torch::Tensor>> data_iter(torch::Tensor X, torch::Tensor Y, int64_t batch_size) {
	int64_t num_examples = X.size(0);
	std::list<std::pair<torch::Tensor, torch::Tensor>> batched_data;
	// data index
	std::vector<int64_t> index;
	for (int64_t i = 0; i < num_examples; ++i) {
		index.push_back(i);
	}
	std::random_shuffle(index.begin(), index.end());

	for (int64_t i = 0; i < index.size(); i +=batch_size) {
		std::vector<int64_t>::const_iterator first = index.begin() + i;
		std::vector<int64_t>::const_iterator last = index.begin() + std::min(i + batch_size, num_examples);
		std::vector<int64_t> indices(first, last);

		int64_t idx_size = indices.size();
		torch::Tensor idx = (torch::from_blob(indices.data(), {idx_size}, torch::kInt64)).clone();

		auto batch_x = X.index_select(0, idx);
		auto batch_y = Y.index_select(0, idx);

		batched_data.push_back(std::make_pair(batch_x, batch_y));
	}
	return( batched_data );
}



