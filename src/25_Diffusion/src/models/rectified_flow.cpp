#include "rectified_flow.h"

#include <utility>

using namespace torch::indexing;

RectifiedFlowImpl::RectifiedFlowImpl(const std::shared_ptr<Module> &model, SamplerOptions &options,
                                     const std::shared_ptr<RectifiedFlowImpl>& last_flow)
        : options(options) {
    this->model = register_module("model", model);
    if (last_flow != nullptr) {
        this->last_flow = register_module("last_flow", last_flow);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RectifiedFlowImpl::p_x(const torch::Tensor &x) {

    // There z0 be gauss, but any other should be ok.
    auto z0 = torch::randn_like(x);
    torch::Tensor z1;
    z1 = x;

    auto ti = torch::rand({z0.size(0), 1, 1, 1}, x.options());
    auto z_t = ti * z1 + (1. - ti) * z0;
    auto target = z1 - z0;

    return std::make_tuple(z_t, ti.squeeze(-1).squeeze(-1), target);
}

torch::Tensor RectifiedFlowImpl::forward(torch::Tensor x, torch::Tensor t_in) {
    auto t_e = t(t_in);
    auto x_r = model->as<Unet>()->forward(std::move(x), t_e);
    return x_r;
}

torch::Tensor RectifiedFlowImpl::sample_ode(const torch::Tensor& z0, int N, bool verbose) {
    torch::NoGradGuard no_grad;
    if (N == -1) {
        N = options.T();
    }
    auto dt = 1. / N;
    auto z = z0;
    auto batch_size = z.size(0);

    for (int i = 0; i < N; ++i) {
        auto ti = torch::ones({batch_size, 1}, z0.options()) * i / N;
        auto pred = forward(z, ti);
        z = z + pred * dt;
        if (verbose) {
            std::printf("\rSample [%1d / %5d] ", i, N);
        }
    }

    return z;
}

torch::Tensor RectifiedFlowImpl::t(const torch::Tensor &ti) {
    return ti.expand({ti.size(0), options.embedding_size()});
}

std::shared_ptr<torch::nn::Module> RectifiedFlowImpl::get_model() {
    return model;
}

std::shared_ptr<Sampler> RectifiedFlowImpl::copy() {
    return std::make_shared<RectifiedFlowImpl>(RectifiedFlowImpl(Unet(*(get_model()->as<Unet>())).ptr(), options));
}

torch::Tensor RectifiedFlowImpl::sample(const std::string &path, torch::Device device) {
    return sample(std::make_shared<std::string>(path), 4, nullptr, 0, device);
}

torch::Tensor RectifiedFlowImpl::sample(const std::string &path, int n, torch::Device device) {
    return sample(std::make_shared<std::string>(path), n, nullptr, 0, device);
}

torch::Tensor RectifiedFlowImpl::sample(torch::Device device) {
    return sample(nullptr, 4, nullptr, 0, device);
}

torch::Tensor RectifiedFlowImpl::sample(const std::shared_ptr<std::string> &path, int n,
                               const std::shared_ptr<torch::Tensor> &z_samples,
                               int t0, torch::Device device) {
    torch::NoGradGuard no_grad;
    torch::Tensor z;

    int img_height = options.img_height(), img_width = options.img_width();
    auto T = options.T();

    if (z_samples == nullptr) {
        z = torch::randn({(long) std::pow(n, 2), 3, img_height, img_width}, torch::TensorOptions().device(device));
    } else {
        z = z_samples->clone();
    }

    z = sample_ode(z, T, true);

    auto x_samples = torch::clip(z, -1, 1); // (n * n, 3, h, w)
    auto result = x_samples.clone();
    x_samples = ((x_samples + 1.) / 2. * 255).detach().to(torch::kU8).cpu().permute(
            {0, 2, 3, 1}).contiguous(); // (n * n, h, w, 3)
    if (path != nullptr) {
        DDPMImpl::save_fig(x_samples, *path, n, img_height, img_width);
    }

    return result;
}
