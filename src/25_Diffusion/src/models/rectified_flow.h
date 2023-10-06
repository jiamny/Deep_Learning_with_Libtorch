#pragma once
#ifndef DIFFUSION_RECTIFIED_FLOW_H
#define DIFFUSION_RECTIFIED_FLOW_H


#include "ddpm.h"

class RectifiedFlowImpl : public Sampler {

private:

    std::shared_ptr<Module> model{nullptr};

    /**
     * Expand t_in to emb size.
     * @param ti
     * @return
     */
    torch::Tensor t(const torch::Tensor& ti);

    std::vector<torch::Tensor> p0;
    std::vector<torch::Tensor> p1;

public:

    RectifiedFlowImpl(const std::shared_ptr<Module> &model,
                      SamplerOptions &options,
                      const std::shared_ptr<RectifiedFlowImpl>& last_flow = nullptr);

    torch::Tensor forward(torch::Tensor x, torch::Tensor t_in) override;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> p_x(const torch::Tensor &x) override;

    torch::Tensor sample_ode(const torch::Tensor& z0, int N = -1, bool verbose=false);

    torch::Tensor
    sample(const std::shared_ptr<std::string>& path = nullptr, int n = 4, const std::shared_ptr<torch::Tensor>& z_samples = nullptr,
           int t0 = 0, torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    torch::Tensor sample(const std::string& path, torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    torch::Tensor sample(const std::string& path, int n, torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    torch::Tensor sample(torch::Device device = torch::Device(torch::kCUDA, 0)) override;

    std::shared_ptr<Sampler> copy() override;

    std::string name() override {
        return "RectifiedFlow";
    }

public:
    SamplerOptions options;
    std::shared_ptr<RectifiedFlowImpl> last_flow {nullptr};

    std::shared_ptr<Module> get_model() override;
};

TORCH_MODULE(RectifiedFlow);


#endif //DIFFUSION_RECTIFIED_FLOW_H
