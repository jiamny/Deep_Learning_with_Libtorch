#include "model_serialize.h"

std::vector<char> get_the_bytes(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

void write_bytes(std::vector<char> bytes, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(bytes.data(), bytes.size());
        file.close();
    }
}

void save_state_dict(const std::shared_ptr<torch::nn::Module>& model, const std::string& pt_pth) {
    const auto &model_params = model->named_parameters();
    torch::Dict<std::string, at::Tensor> weights;
    for (const auto& pair : model_params) {
        weights.insert(pair.key(), pair.value().detach());
    }
    std::vector<char> f = torch::pickle_save(weights);
    write_bytes(f, pt_pth);
}


void load_state_dict(const std::shared_ptr<torch::nn::Module>& model, const std::string& pt_pth) {
    std::vector<char> f = get_the_bytes(pt_pth);

    c10::Dict<torch::IValue, torch::IValue> weights = torch::pickle_load(f).toGenericDict();

    const torch::OrderedDict<std::string, at::Tensor> &model_params = model->named_parameters();
    std::vector<std::string> param_names;
    for (auto const &w: model_params) {
        param_names.push_back(w.key());
    }

    torch::NoGradGuard no_grad;
    for (auto const &w: weights) {
        std::string name = w.key().toStringRef();
        at::Tensor param = w.value().toTensor();

        if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()) {
            model_params.find(name)->copy_(param);
        } else {
            std::cout << name << " does not exist among model parameters." << std::endl;
        };

    }
}