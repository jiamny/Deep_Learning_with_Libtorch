//
//  main.cpp
//  DCGAN
//
//  Created by Kushashwa Ravi Shrimali on 19/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include <iostream>
#include <torch/torch.h>
#include "network.hpp"
#include "dataset.hpp"

class Arguments {
  public:
    std::string dataroot;
    int workers;
    int batch_size;
    int image_size;
    int nc;
    int nz;
    int ngf;
    int ndf;
    int num_epochs;
    float lr;
    float beta1;
    int ngpu;
    Arguments(std::string data_root, int num_workers, int bs, int img_size, int num_channels, int length_latent_vector, int depth_feature_maps_g, int depth_feature_maps_d, int number_epochs, float learning_rate, float beta_1, int num_gpu) {
      dataroot = data_root;
      workers = num_workers;
      batch_size = bs;
      image_size = img_size;
      nc = num_channels;
      nz = length_latent_vector;
      ngf = depth_feature_maps_g;
      ndf = depth_feature_maps_d;
      num_epochs = number_epochs;
      lr = learning_rate;
      beta1 = beta_1;
      ngpu = num_gpu;
    };
};

int ngf = 64;
int ndf = 64;

int main(int argc, const char * argv[]) {
  Arguments args = Arguments("/media/stree/localssd/DL_data", 2, 64, 64, 3, 300, 64, 64, 3, 0.001, 0.5, 1);
  std::string images_name = args.dataroot + "/CelebA/train/";

  std::vector<std::string> folders_name;
  folders_name.push_back(images_name);

  // Get paths of images and labels as int from the folder paths
  std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder(folders_name);
  std::vector<std::string> list_images = pair_images_labels.first;
  std::vector<int> list_labels = pair_images_labels.second;
  /*
  for( int i =0; i < list_images.size(); i++ ) {
	  std::cout << "img = " << list_images.at(i) << " label = " << list_labels.at(i) << std::endl;
  }
  */

  // auto custom_dataset2 = CustomDataset(list_images, list_labels, 64);
  // custom_dataset2.show_batch(4);
  // Originally 224 size, resize to 64 instead 
  auto custom_dataset_init = CustomDataset(list_images, list_labels, 64);
  custom_dataset_init.show_batch(5);
  custom_dataset_init.show_sample(5);

  auto custom_dataset = custom_dataset_init.map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),
		  torch::data::DataLoaderOptions().batch_size(args.batch_size).workers(2));

  torch::Device device = torch::kCPU;
  if(torch::cuda::is_available()) {
    device = torch::kCUDA;
  }

  std::cout << "Using device: " << device << std::endl;
  Generator G = Generator(/*nc_=*/3, /*nz_=*/300, /*ngf_=*/64);
  Discriminator D = Discriminator(/*nc_=*/3, /*ndf_=*/64);

  torch::nn::Sequential netG = G.get_module();
  torch::nn::Sequential netD = D.get_module();

  netG->to(device);
  netD->to(device);

  torch::optim::Adam optimizerG(
      netG->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));
  torch::optim::Adam optimizerD(
      netD->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));

  int printEveryCheckpoint = 20;
  bool restoreFromCheckpoint = false; // set to false if you don't want to restore from checkpoint saved earlier

  if( restoreFromCheckpoint ) {
    std::cout << "restoring from checkpoint..." << std::endl;
    torch::load(netG, "models/DCGAN_celebA_generator-checkpoint.pt");
    torch::load(optimizerG, "models/DCGAN_celebA_generator-optimizer-checkpoint.pt");
    torch::load(netD, "models/DCGAN_celebA_discriminator-checkpoint.pt");
    torch::load(optimizerD, "models/DCGAN_celebA_discriminator-optimizer-checkpoint.pt");
    std::cout << "restoring done" << std::endl;
  }

  int64_t checkpoint_counter = 1;
  auto options = torch::TensorOptions().device(device).requires_grad(false);
  for(int64_t epoch=1; epoch<=20; ++epoch) {
    int64_t batch_index=0;

    for(torch::data::Example<>& batch: *data_loader) {

     /*
      * Warning: An output with one or more elements was resized since it had shape [64, 1, 1, 1],
      * which does not match the required output shape [64, 1, 1, 64]. This behavior is deprecated,
      * and in a future PyTorch release outputs will not be resized unless they have zero elements.
      * You can explicitly reuse an out tensor t by resizing it, inplace,
      * to zero elements with t.resize_(0).
      *
      * Call .reshape({batch_size}) on results of the forward calls.
      */
      const auto batch_size = batch.data.size(0);

      netD->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(1.0, 1.0);

      torch::Tensor real_output = netD->forward(real_images).reshape({batch_size});

      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images
      torch::Tensor noise = torch::randn({batch.data.size(0), args.nz, 1, 1}, device);
      torch::Tensor fake_images = netG->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);

      torch::Tensor fake_output = netD->forward(fake_images.detach()).reshape({batch_size});

      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      optimizerD.step();

      // Train generator
      netG->zero_grad();
      fake_labels.fill_(1);

      fake_output = netD->forward(fake_images).reshape({batch_size});

      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      optimizerG.step();

      std::cout << "Epoch: " << epoch << ", D_loss: " << d_loss.item<float>() << ", G_loss: " << g_loss.item<float>() << std::endl;
      batch_index++;
      // check point for every printEveryCheckpoint batches
      if(batch_index % printEveryCheckpoint == 0) {
        torch::save(netG, "models/DCGAN_celebA_generator-checkpoint.pt");
        torch::save(optimizerG, "models/DCGAN_celebA_generator-optimizer-checkpoint.pt");
        torch::save(netD, "models/DCGAN_celebA_discriminator-checkpoint.pt");
        torch::save(optimizerD, "models/DCGAN_celebA_discriminator-optimizer-checkpoint.pt");
        torch::Tensor samples = netG->forward(torch::randn({64, args.nz, 1, 1}, options));
        torch::save(samples, torch::str("models/DCGAN_celebA_dcgan-sample-", ++checkpoint_counter, ".pt"));
        std::cout << "\n-> checkpoint " << ++checkpoint_counter << "\n";
      }
    }
  }

  std::cout << "Done!\n";
  return 0;
}
