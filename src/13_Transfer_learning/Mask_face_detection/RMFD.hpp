#pragma once

#include "utils.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
class RMFD : public torch::data::datasets::Dataset<RMFD> {
private:

    std::vector<torch::Tensor> images, labels;
    size_t ds_size;
    void load_data(const std::string& folderName, const int label);

public:
    // Constructor
    RMFD(const std::string& rawFaceFolder, const std::string& maskedFaceFolder);

    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;

    // Returns the length of data
    torch::optional<size_t> size() const override;
};
