#include <torch/torch.h>
#include "train.h"

#include <cmath>
#include <cstdio>
#include <iostream>
using namespace torch;

struct DCGANGeneratorImpl : nn::Module {
  DCGANGeneratorImpl(int noise_size)
      : conv1(nn::ConvTranspose2dOptions(noise_size, 256, 4)
                  .bias(false)),
        batch_norm1(256),
        conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm2(128),
        conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm3(64),
        conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false))
 {
   // register_module() is needed if we want to use the parameters() method later on
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("conv3", conv3);
   register_module("conv4", conv4);
   register_module("batch_norm1", batch_norm1);
   register_module("batch_norm2", batch_norm2);
   register_module("batch_norm3", batch_norm3);
 }

 torch::Tensor forward(torch::Tensor x) {
   x = torch::relu(batch_norm1(conv1(x)));
   x = torch::relu(batch_norm2(conv2(x)));
   x = torch::relu(batch_norm3(conv3(x)));
   x = torch::tanh(conv4(x));
   return x;
 }

 nn::ConvTranspose2d conv1, conv2, conv3, conv4;
 nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);

int torchJob::train() {
  torch::manual_seed(1);

  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }else{
    std::cout << "CUDA is not available! Exiting." << std::endl;
    exit(0);
  }

  DCGANGenerator generator(noise_size);
  generator->to(device);

  nn::Sequential discriminator(
      // Layer 1
      nn::Conv2d(
          nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 2
      nn::Conv2d(
          nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(128),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 3
      nn::Conv2d(
          nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(256),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 4
      nn::Conv2d(
          nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
      nn::Sigmoid());
  discriminator->to(device);

  // Assume the MNIST dataset is available under `kDataFolder`;
  auto dataset = torch::data::datasets::MNIST(data_folder)
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                     .map(torch::data::transforms::Stack<>());
  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(batch_size));

  auto data_loader = torch::data::make_data_loader(std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));
  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));
  if (restore) {
    torch::load(generator, "generator-checkpoint.pt");
    torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::load(discriminator, "discriminator-checkpoint.pt");
    torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  }
  int64_t checkpoint_counter = 1;
  for (int64_t epoch = 1; epoch <= n_epochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty({batch.data.size(0),1,1,1}, device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real =
          torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise =
          torch::randn({batch.data.size(0), noise_size, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros({batch.data.size(0),1,1,1}, device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      batch_index++;
      if (batch_index % log_interval == 0) {
        std::printf(
            "\r[%2ld/%d][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
            epoch,
            n_epochs,
            batch_index,
            batches_per_epoch,
            d_loss.item<float>(),
            g_loss.item<float>());
      }

      if (batch_index % check_point_every == 0) {
        // Checkpoint the model and optimizer state.
        torch::save(generator, "generator-checkpoint.pt");
        torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::save(discriminator, "discriminator-checkpoint.pt");
        torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
        // Sample the generator and save the images.
        torch::Tensor samples = generator->forward(torch::randn(
            {n_sample_per_check_point, noise_size, 1, 1}, device));
        torch::save((samples + 1.0) / 2.0,
            torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
        std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
      }
    }
  }

  std::cout << "Training complete!" << std::endl;
  return 0;
}
