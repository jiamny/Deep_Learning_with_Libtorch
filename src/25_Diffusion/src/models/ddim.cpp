#include <ATen/autocast_mode.h>
#include "ddim.h"

using namespace torch::indexing;

DDIMImpl::DDIMImpl(const std::shared_ptr<Module> &model, SamplerOptions &options)
        : DDPMImpl(model, options) {

    stride = options.stride();
    bar_alpha_ = bar_alpha.index({Slice({None, None, stride})});
    bar_alpha_pre_ = torch::pad(bar_alpha_.index({Slice(None, -1)}), {1, 0}, "constant", 1);
    bar_beta_ = torch::sqrt(1.0 - torch::pow(bar_alpha_, 2));
    bar_beta_pre_ = torch::sqrt(1.0 - torch::pow(bar_alpha_pre_, 2));
    alpha_ = bar_alpha_ / bar_alpha_pre_;
    sigma_ = bar_beta_pre_ / bar_beta_ * torch::sqrt(1.0 - torch::pow(alpha_, 2)) * options.eta();
    epsilon_ = bar_beta_ - alpha_ * torch::sqrt(torch::pow(bar_beta_pre_, 2) - torch::pow(sigma_, 2));

    register_buffer("bar_alpha_", bar_alpha_);
    register_buffer("bar_alpha_pre_", bar_alpha_pre_);
    register_buffer("bar_beta_", bar_beta_);
    register_buffer("bar_beta_pre_", bar_beta_pre_);
    register_buffer("alpha_", alpha_);
    register_buffer("sigma_", sigma_);
    register_buffer("epsilon_", epsilon_);
}

std::shared_ptr<Sampler> DDIMImpl::copy() {
    return std::make_shared<DDIMImpl>( DDIMImpl(Unet(*(get_model()->as<Unet>())).ptr(), options));
}

torch::Tensor DDIMImpl::sample(const std::string &path, torch::Device device) {
    return sample(std::make_shared<std::string>(path), 4, nullptr, 0, device);
}

torch::Tensor DDIMImpl::sample(const std::string &path, int n, torch::Device device) {
    return sample(std::make_shared<std::string>(path), n, nullptr, 0, device);
}

torch::Tensor DDIMImpl::sample(torch::Device device) {
    return sample(nullptr, 4, nullptr, 0, device);
}

torch::Tensor
DDIMImpl::sample(const std::shared_ptr<std::string>& path, int n, const std::shared_ptr<torch::Tensor>& z_samples,
       int t0 , torch::Device device) {
    torch::NoGradGuard no_grad;

    int img_height, img_width;
    std::tie(img_height, img_width) = img_size;

    auto T_ = bar_alpha_.size(0);

    torch::Tensor z;
    if (z_samples == nullptr) {
        z = torch::randn({(long) std::pow(n, 2), 3, img_height, img_width}, torch::TensorOptions().device(device));
    } else {
        z = z_samples->clone();
    }

    at::autocast::set_enabled(true);
    for (int i = t0; i < T_; i++) {
        auto t = T_ - i - 1;
        auto bt = torch::tensor({t * stride}, torch::TensorOptions().device(device)).repeat(z.size(0));
        z = z - epsilon_.index({t}) * forward(z, bt);
        z = z / alpha_.index({t});
        z = z + torch::randn_like(z) * sigma_.index({t});

        std::printf("\rSample [%1d / %5zd] ", i, T_);
    }
    at::autocast::clear_cache();
    at::autocast::set_enabled(false);

    auto x_samples = torch::clip(z, -1, 1); // (n * n, 3, h, w)
    auto result = x_samples.clone();
    x_samples = ((x_samples + 1.) / 2. * 255).detach().to(torch::kU8).cpu().permute(
            {0, 2, 3, 1}).contiguous(); // (n * n, h, w, 3)
    if (path != nullptr) {
        save_fig(x_samples, *path, n, img_height, img_width);
    }

    return result;

}