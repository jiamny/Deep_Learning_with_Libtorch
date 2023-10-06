#include "trainer.h"
#include "utils/ema.h"
#include "utils/readfile.h"
#include "utils/path.h"
#include "datasets/folder.h"
#include "utils/model_serialize.h"
#include <random>
#include <utility>
#include <ATen/autocast_mode.h>

Trainer::Trainer(std::shared_ptr<Sampler> sampler,
                 const std::tuple<int, int>& img_size,
                 std::string &exp_name,
                 int train_batch_size,
                 double train_lr,
                 int train_num_epochs,
                 double ema_decay,
                 int num_workers,
                 int save_and_sample_every,
                 int accumulation_steps,
                 bool amp_enable) {

    this->sampler = std::move(sampler);
    this->img_size = img_size;
    this->train_batch_size = train_batch_size;
    this->train_lr = train_lr / accumulation_steps; // auto scale lr by accumulation_steps
    this->train_num_epochs = train_num_epochs;
    this->ema_decay = ema_decay;
    this->num_workers = num_workers;
    this->save_and_sample_every = save_and_sample_every;
    this->accumulation_steps = accumulation_steps;
    this->amp_enable = amp_enable;

    sample_path = std::string("experiments").append({file_sepator()}).append(exp_name).append({file_sepator()}).append(
            "outputs");
    checkpoint_path = std::string("experiments").append({file_sepator()}).append(exp_name).append(
            {file_sepator()}).append("checkpoints");

    // make experiments save path
    std::cout << "checkpoint_path: " << checkpoint_path << '\n';
    std::cout << "sample_path: " << sample_path << '\n';
    makedirs(checkpoint_path.c_str());
    makedirs(sample_path.c_str());

    // Get model device.
    device = this->sampler->parameters(true).back().device();

    // make a new sampler as shadow.
    sampler_shadow = this->sampler->copy();
    sampler_shadow->to(device);

    // Init copy
    update_average(sampler_shadow, this->sampler, 0.);

}

void Trainer::train(std::string dataset_path, std::string image_type) {
    auto dataset = ImageFolderDataset(std::move(dataset_path), img_size, std::move(image_type))
            // RandomFliplr
            .map(torch::data::transforms::Lambda<torch::data::TensorExample>([](torch::data::TensorExample input) {
                if ((torch::rand(1).item<double>() < 0.5)) {
                    input.data = torch::flip(input.data, {-1});
                }
                return input;
            }))
            .map(torch::data::transforms::Stack<torch::data::TensorExample>());

    const size_t dataset_size = dataset.size().value();
    const size_t num_step = dataset_size / train_batch_size;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions()
                    .batch_size(train_batch_size)
                    .workers(num_workers)
                    .drop_last(true)
    );

    torch::optim::AdamW optimizer(sampler->parameters(), train_lr);
    auto scheduler = torch::optim::StepLR(optimizer, 3, 0.999);

    long step;
    step = 0;
    for (size_t epoch = 1; epoch <= train_num_epochs; ++epoch) {

        size_t batch_idx = 0;
        sampler->zero_grad(true);
        for (auto &batch: *train_loader) {

            if (step % save_and_sample_every == 0) {
                auto exp_img_ema_path = (std::stringstream() << sample_path << "/" << sampler->name() << "_ckpt_" << epoch << "_" << step
                                                         << "_ema.png").str();
                auto exp_img_path = (std::stringstream() << sample_path << "/" << sampler->name() << "_ckpt_" << epoch << "_" << step
                                                         << ".png").str();
                sampler_shadow->sample(exp_img_ema_path, 4);
                sampler->sample(exp_img_path, 4);
                auto export_path = (std::stringstream() << checkpoint_path << "/" << sampler->name() << "_ckpt_" << epoch << "_" << step
                                                        << ".pt").str();
                auto export_py_path = export_path + "h";
//                torch::save(sampler_shadow->get_model(), export_path);
                save_state_dict(sampler_shadow->get_model(), export_py_path);
            }


            auto data = batch.data.to(device);

            if (amp_enable) {
                at::autocast::set_enabled(true);
            }
            torch::Tensor noise_images;
            torch::Tensor steps;
            torch::Tensor noise;
            std::tie(noise_images, steps, noise) = sampler->p_x(data);

            auto denoise = sampler->forward(noise_images, steps);
            auto loss = torch::sum((denoise - noise).pow(2), {1, 2, 3}, true).mean();

            if (amp_enable) {
                at::autocast::clear_cache();
                at::autocast::set_enabled(false);
            }

            loss.backward();

            if ((step + 1) % accumulation_steps == 0) {
                optimizer.step();
                sampler->zero_grad(true);
                update_average(sampler_shadow, sampler, ema_decay);
            }

            std::printf("\rEpoch [%1zd / %5d] .. [%5zd/%5zd] Loss: %.4f",
                        epoch,
                        train_num_epochs,
                        batch_idx,
                        num_step,
                        loss.template item<float>());

            step += 1;
            batch_idx += 1;
        }

        scheduler.step();
    }
}
