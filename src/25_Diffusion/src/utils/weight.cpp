#include <functional>
#include <torch/torch.h>

std::function<void(torch::nn::Module &)> weights_norm_init(double var_scale = 0.02) {
    auto _ = [var_scale](const torch::nn::Module &module) {
        torch::NoGradGuard no_grad;
        auto classname = module.name();
        if (classname.find("Conv2d") != classname.npos) {
            torch::nn::init::normal_(module.as<torch::nn::Conv2d>()->weight, 0.0, var_scale);
        } else if (classname.find("ConvTranspose2d") != classname.npos) {
            torch::nn::init::normal_(module.as<torch::nn::ConvTranspose2d>()->weight, 0.0, var_scale);
        } else if (classname.find("BatchNorm") != classname.npos) {
            auto m = module.as<torch::nn::BatchNorm2d>();
            if (m->options.affine()) {
                torch::nn::init::normal_(m->weight, 1.0, var_scale);
                torch::nn::init::constant_(m->bias, 0.);
            }
        }
    };
    return _;
}