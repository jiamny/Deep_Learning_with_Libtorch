#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "src/trainer.h"
#include "src/datasets/folder.h"
#include "src/utils/weight.h"
#include "src/utils/model_serialize.h"
#include "src/utils/hand.h"
#include "CLI11.hpp"
#include "src/models/ddim.h"
#include "src/models/rectified_flow.h"

// --------------------------------------------------------------------------------------------
// Download dataset from: https://www.kaggle.com/datasets/splcher/animefacedataset/data
// --------------------------------------------------------------------------------------------

using namespace cv;

// this function use to test the Unet model's validity.
void test_unet_train(Unet &model,
                     std::string dataset_path,
                     std::tuple<int, int> img_size,
                     int train_batch_size,
                     int num_workers,
                     int epochs = 100,
                     torch::Device device = torch::Device(torch::kCPU)) {
    auto dataset = ImageFolderDataset(dataset_path, img_size)
            .map(torch::data::transforms::Stack<torch::data::TensorExample>());

    const size_t dataset_size = dataset.size().value();
    const size_t num_step = dataset_size / train_batch_size;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions()
                    .drop_last(true)
                    .batch_size(train_batch_size)
                    .workers(num_workers));

    torch::optim::AdamW optimizer(model->parameters(), 2e-4);

    auto step = 0;
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        size_t batch_idx = 0;

        for (auto &batch: *train_loader) {
            auto images = batch.data.to(device);
            auto steps = torch::zeros({images.size(0), 64},
                                      torch::TensorOptions().dtype(images.dtype()).device(device));

            optimizer.zero_grad();
            auto denoise = model->forward(images + 0.5 * torch::randn_like(images), steps);
            auto loss = torch::sum((denoise - images).pow(2), {1, 2, 3}, true).mean();

            loss.backward();
            optimizer.step();

            std::printf("\rEpoch [%1zd / %5d] .. [%5zd/%5zd] Loss: %.4f",
                        epoch,
                        epochs,
                        batch_idx,
                        num_step,
                        loss.template item<float>());
            batch_idx += 1;

            if ((step + 1) % 50 == 0) {
                auto t = denoise.detach().cpu().index({0}); // (h, w, 3)
                t = t.permute({1, 2, 0}).contiguous();
                t = (t + 1) / 2. * 255;
                t = t.to(torch::kU8);
                cv::Mat mat = cv::Mat(128, 128, CV_8UC3, t.data_ptr());
                cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
                cv::imwrite((std::stringstream() << "test_" << step << ".png").str(), mat);
            }

            step += 1;
        }
    }
}


auto main(int argc, char **argv) -> int {

    at::globalContext().setBenchmarkCuDNN(true);

    /*------------define model scale.------------*/
    int img_width = 128;
    int img_height = 128;
    std::tuple<int, int> img_size({img_height, img_width});     // image size (height, width)
    std::vector<int> scales = {1, 1, 2, 2, 4, 4};				// scale size for each block, base on emb_dim
    int emb_dim = 64;                                // base channels dim
    int T = 100;                                     // sampler steps

    // auto device = torch::Device(torch::kCUDA, 0); // indicate training/testing on which device
    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    int stride = 8;                                  // Sample steps = T // stride. E.g, 1000 / 50 = 20
    int attn_resolution = 16;
    bool rectify = false;      						 // when using `rflow` sampler, indicate whether to rectify.
    /*-------------------------------------------*/

    /*------------define train/test params--------*/
    std::string mode("test");
    std::string exp_name("AnimeFace");
    std::string sampler_type = "ddim";
    std::string dataset_path("/media/hhj/localssd/DL_data/AnimeFace/images");
    std::string image_suffix("jpg");
    std::string pretrained_weights;

    // path to trained weight file
    std::string weight_path("experiments/AnimeFace/checkpoints/DDIM_ckpt_98_1950.pth");

    int batch_size = 16;
    double learning_rate = 2e-4;
    int num_epochs = 1000;
    double ema_decay = 0.995;
    int num_workers = 4;
    int save_and_sample_every = 500;
    int accumulation_steps = 1;
    bool amp_enable = false;

    /*--------------------------------------------*/

    CLI::App app{"TorchLib Diffusion model implementation"};
    app.add_option("-m,--mode", mode, "train/test. Default. train");
    app.add_option("-d,--dataset", dataset_path, "dataset path");
    app.add_option("-b,--batch_size", batch_size, "batch size");
    app.add_option("--lr,--learning_rate", learning_rate, "learning rate");
    app.add_option("--epochs", num_epochs, "number of epochs.");
    app.add_option("--n_cpu", num_workers, "number of cpu threads to use during batch generation");
    app.add_option("--name", exp_name, "experiment name");
    app.add_option("-p,--pretrained_weights", pretrained_weights, "pretrained model path");
    app.add_option("-w,--weight_path", weight_path, "weight_path for inference");
    app.add_option("-t,--type", sampler_type, "sampler type [ddpm/ddim/rflow], Default, ddim");
    app.add_option("-s,--stride", stride, "sample stride for ddim");
    app.add_option("--amp,--amp_enable", amp_enable, "whether enable amp autocast, Default, false.");
    app.add_option("--accum,--accumulation", accumulation_steps, "accumulation steps, Default, 1.");
    app.add_option("-r,--rectify", rectify, "when using `rflow` sampler, indicate whether to rectify. Default, false");
    app.add_option("-n,--num_steps", T, "sample times. Default, 1000");
    app.add_option("-x,--suffix", image_suffix, "image suffix, Default, jpg");
    CLI11_PARSE(app, argc, argv);

    auto unet_options = UnetOptions(img_height, img_width, scales)
            .emb_dim(emb_dim)
            .attn_resolution(attn_resolution)
            .img_c(3);
    auto model = Unet(unet_options);

    bool load_pretrained_weights = false;

    if( ! weight_path.empty() || ! pretrained_weights.empty() ) {
    	 try {
    	        // load pytorch trained model.
    	        auto load_path = ! weight_path.empty() ? weight_path : pretrained_weights;
    	        if(endswith(load_path, "pth")) {
    	            load_state_dict(model.ptr(), load_path);
    	        } else {
    	            torch::load(model, load_path);
    	        }
    	        std::cout << "Load weight " << load_path << " successful!" << std::endl;
    	        load_pretrained_weights = true;
    	    } catch (std::exception &e) {
    	        std::cout << e.what() << std::endl;
    	    }
    }

    auto sampler_options = SamplerOptions()
            .img_width(img_width)
            .img_height(img_height)
            .T(T)
            .embedding_size(emb_dim)
            .stride(stride);

    std::cout << "Sampler type: " << sampler_type << std::endl;
    std::shared_ptr<Sampler> diffusion{nullptr};
    if (toLowerCase(sampler_type) == "ddim") {
        diffusion = std::dynamic_pointer_cast<Sampler>(DDIM(model.ptr(), sampler_options).ptr());
    } else if (toLowerCase(sampler_type) == "ddpm") {
        diffusion = std::dynamic_pointer_cast<Sampler>(DDPM(model.ptr(), sampler_options).ptr());
    } else if (toLowerCase(sampler_type) == "rflow") {
        diffusion = std::dynamic_pointer_cast<Sampler>(RectifiedFlow(model.ptr(), sampler_options).ptr());
        if (rectify) {
            diffusion = std::dynamic_pointer_cast<Sampler>(RectifiedFlow(model.ptr(), sampler_options,
                                                                         std::dynamic_pointer_cast<RectifiedFlowImpl>(
                                                                                 diffusion)).ptr());
        }
    } else {
        throw std::invalid_argument("Unsupported sampler type");
    }

    if(mode == "test") {
        std::cout << "Running at inference mode..." << std::endl;
        try {
            diffusion->to(device);
            diffusion->eval();

            int64 start = cv::getTickCount();
            auto x_samples = diffusion->sample("./demo.png", device);
            double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
            std::cout << duration / T << " s per prediction" << std::endl;
        } catch (const std::exception &e) {
            std::cout << e.what() << std::endl;
            return -1;
        }
    } else {
        try {
            std::cout << "Running at training mode..." << std::endl;
            std::cout << "Experiments name: " << (exp_name.empty() ? "(Empty)" : exp_name) << std::endl;
            if (!load_pretrained_weights) {
                diffusion->apply(weights_norm_init());
            }

            diffusion->to(device);
            auto trainer = Trainer(diffusion,
                    /*img_size=*/img_size,
                    /*exp_name=*/exp_name,
                    /*train_batch_size=*/batch_size,
                    /*train_lr=*/learning_rate,
                    /*train_num_epochs=*/num_epochs,
                    /*ema_decay=*/ema_decay,
                    /*num_workers=*/num_workers,
                    /*save_and_sample_every=*/save_and_sample_every,
                    /*accumulation_steps=*/accumulation_steps,
                    /*amp_enable=*/amp_enable
            );
            trainer.train(dataset_path, image_suffix);
            std::cout << '\n';
        } catch (std::exception &e) {
            std::cout << e.what() << std::endl;
            return -1;
        }
    }
    std::cout << "Done!\n";
    return 0;
}
