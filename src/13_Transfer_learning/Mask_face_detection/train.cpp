#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

#include "utils.hpp"
#include "RMFD.hpp"

using namespace std;


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear& lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size) {
    float best_accuracy = 0.0f;

    net.eval();

    for(int epoch=0; epoch<5; epoch++) {
        float mse = 0.0f;
        float Acc = 0.0f;

        int batch_index = 0;

        for(auto& batch: *data_loader) {
            auto data = batch.data;
            auto target = batch.target.squeeze();

            optimizer.zero_grad();

            vector<torch::jit::IValue> input;
            input.push_back(data);
            auto output = net.forward(input).toTensor();

            output = output.view({output.size(0), -1});
            output = lin(output);

            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);

            loss.backward();
            optimizer.step();

            auto acc = output.argmax(1).eq(target).sum();

            Acc += acc.template item<float>();
            mse += loss.template item<float>();

            batch_index++;
        }

        float MSE = mse/batch_index; // Take mean of loss
        float Accuracy =  Acc/dataset_size;

        cout << "Accuracy: " << Accuracy << ", " << "MSE: " << MSE << endl;

        if (Accuracy > best_accuracy) {
            best_accuracy = Accuracy;
            cout << "Saving model" << endl;
            torch::save(lin, "./models/mask_face_detection_model_linear.pt");
        }
    }
}


//----------------------------------------------------------------------------
// download Face-Mask dataset from:
//
//		https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
//
//----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    //if (argc!=4)
    //    throw std::runtime_error("Usage: ./exe rawFaceFolder maskedFaceFolder modelWithoutLastLayer");

    string rawFaceFolder = "./data/Face-Mask/without_mask";		//argv[1];

    string maskedFaceFolder = "./data/Face-Mask/with_mask";		//argv[2];

    cout << rawFaceFolder << endl;
    cout << maskedFaceFolder << endl;

    auto train_dataset = RMFD(rawFaceFolder, maskedFaceFolder).map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_dataset.size().value();
    cout << "train_dataset_size: " << train_dataset_size << endl;

    cout << train_dataset_size << endl;

    auto model = torch::jit::load("./models/Transfer_learning/resnet18_without_last_layer.pt"); 	//argv[3]);

    torch::nn::Linear linear_layer(512, 2);

    torch::optim::Adam optimizer(linear_layer->parameters(), torch::optim::AdamOptions(1e-3));

    int batch_size = 4;
    auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
    							torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));

    train(model, linear_layer, train_data_loader, optimizer, train_dataset_size);

    cout << "Done!\n";
    return 0;
}
