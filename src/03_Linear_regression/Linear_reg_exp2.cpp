#include <torch/torch.h>
#include <iostream>


int batch_size = 1;
float learning_rate = 0.001;

int main() {
    std::vector<float> v = {
        2.1, 3.3, 3.6, 4.4, 5.5, 6.3, 6.5, 7.0, 7.5, 9.7, // x
        1.0, 1.2, 1.9, 2.0, 2.5, 2.5, 2.2, 2.7, 3.0, 3.6  //y
    };

    std::vector<float> v2 = {
        2.1, 1.0, 3.3, 1.2, 3.6, 1.9, 4.4, 2.0, 5.5, 2.5,
		6.3, 2.5, 6.5, 2.3, 7.0, 2.7, 7.5, 3.0, 9.7, 3.6// x, y
    }; 

    torch::Tensor data_tensor = torch::from_blob(v2.data(), {10, 2}, torch::kFloat32);
    std::cout << "---\n";
    std::cout << data_tensor.data().sizes() << '\n';

    auto dataset = torch::data::datasets::TensorDataset(data_tensor); //.transpose(0, 1));
    std::cout << dataset.size() << '\n';
    auto data_loader = torch::data::make_data_loader(dataset, batch_size);


    torch::nn::Linear model(1, 1);
    torch::optim::SGD optimizer(model->parameters(), learning_rate);

    for (size_t epoch = 0; epoch < 50; ++epoch) {
        float epoch_loss = 0;
        for (auto&& batch : *data_loader) {
        	std::cout << batch[0].data.sizes() << '\n';
            auto x = torch::tensor(batch[0].data[0].item<float>()); // x
            auto y = torch::tensor(batch[0].data[1].item<float>()); // y

            auto output = model->forward(x);
            auto loss = torch::mse_loss(output, y);
            
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            epoch_loss += loss.item<float>();
        }

        epoch_loss /= 10;
        std::cout << "loss: " << epoch_loss << std::endl;
    }
}

