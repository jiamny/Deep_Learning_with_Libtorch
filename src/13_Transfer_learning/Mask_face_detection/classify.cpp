#include "utils.hpp"
#include <torch/script.h>
#include <chrono>


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
int classify_image(const std::string& imageName, torch::jit::script::Module& model, torch::nn::Linear& linear_layer) {
    torch::Tensor img_tensor = read_image(imageName);
    img_tensor.unsqueeze_(0);

    std::vector<torch::jit::IValue> input;
    input.push_back(img_tensor);
    torch::Tensor temp = model.forward(input).toTensor();

    temp = temp.view({temp.size(0), -1});
    temp = linear_layer(temp);

    temp = temp.argmax(1);

    return *temp.data_ptr<long>();
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    //if (argc!=4)
    //    throw std::runtime_error("Usage: ./exe imageName modelWithoutLastLayer trainedLinearLayer");

	std::string imgF = "./src/13_Transfer_learning/data/person_NM.jpg";

	// Load the model.
    torch::jit::script::Module model;
    model = torch::jit::load("./models/Transfer_learning/resnet18_without_last_layer.pt");
    model.eval();

    torch::nn::Linear linear_layer(512, 2);
    torch::load(linear_layer,
    		"./models/mask_face_detection_model_linear.pt");	//argv[3]);

    auto t1 = std::chrono::high_resolution_clock::now();

    int result = classify_image(imgF, model, linear_layer);	// argv[1]

    auto t2 = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << "Inference time: " << duration << " ms" << std::endl;

    cv::Mat img = cv::imread(imgF);

    if(result==0) {
        std::cout << "No Mask!!" << std::endl;
        cv::namedWindow("No Mask!!", cv::WINDOW_NORMAL);
        cv::imshow("No Mask!!",img);
    } else {
        std::cout << "Has Mask :)" << std::endl;
        cv::namedWindow("Has Mask :)", cv::WINDOW_NORMAL);
        cv::imshow("Has Mask :)",img);
    }

    cv::waitKey(0);

    std::cout << "Done!\n";
    return 0;
}
