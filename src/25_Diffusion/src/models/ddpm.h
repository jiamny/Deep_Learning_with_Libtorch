#pragma once
#include <iostream>
#include <torch/torch.h>

#include "unet.h"

struct SamplerOptions {

    TORCH_ARG(int, img_height);
    TORCH_ARG(int, img_width);
    TORCH_ARG(int, T) = 1000;
    TORCH_ARG(int, embedding_size) = 2;
    TORCH_ARG(int, stride) = 2;
    TORCH_ARG(float, eta) = 1.0;
};

class Sampler : public torch::nn::Module {

public:

    virtual torch::Tensor forward(torch::Tensor x, torch::Tensor t_in) = 0;

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> p_x(const torch::Tensor& x) = 0;

    virtual torch::Tensor
    sample(const std::shared_ptr<std::string>& path = nullptr, int n = 4, const std::shared_ptr<torch::Tensor>& z_samples = nullptr,
           int t0 = 0, torch::Device device = torch::Device(torch::kCUDA, 0)) = 0;

    virtual torch::Tensor sample(const std::string& path, torch::Device device = torch::Device(torch::kCUDA, 0)) = 0;

    virtual torch::Tensor sample(const std::string& path, int n, torch::Device device = torch::Device(torch::kCUDA, 0)) = 0;

    virtual torch::Tensor sample(torch::Device device = torch::Device(torch::kCUDA, 0)) = 0;

    virtual std::shared_ptr<Sampler> copy() = 0;

    virtual std::shared_ptr<Module> get_model() = 0;

    virtual std::string name() = 0;
};

class DDPMImpl : public Sampler {
private:

    std::shared_ptr<Module> model{nullptr};
protected:
    std::tuple<int, int> img_size;
public:
    SamplerOptions options;
    int T;
    torch::Tensor alpha;
    torch::Tensor beta;
    torch::Tensor sigma;
    torch::Tensor bar_alpha;
    torch::Tensor bar_beta;
    torch::Tensor t;
    int embedding_size;
    std::vector<int> steps;

    DDPMImpl(const std::shared_ptr<Module> &model, SamplerOptions& options);

    torch::Tensor forward(torch::Tensor x, torch::Tensor t_in) override;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> p_x(const torch::Tensor& x) override;

    torch::Tensor
    sample(const std::shared_ptr<std::string>& path = nullptr, int n = 4, const std::shared_ptr<torch::Tensor>& z_samples = nullptr,
           int t0 = 0, torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    torch::Tensor sample(const std::string& path, torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    torch::Tensor sample(const std::string& path, int n, torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    torch::Tensor sample(torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    std::shared_ptr<Module> get_model() override;

    std::shared_ptr<Sampler> copy() override;

    static void save_fig(const torch::Tensor &x_samples, std::string &path, int n, int img_height, int img_width);

    std::string name() override {
        return "DDPM";
    }
};

TORCH_MODULE(DDPM);