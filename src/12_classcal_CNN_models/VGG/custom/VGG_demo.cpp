
#include "Classification.h"

int main(int argc, char *argv[]) {
	// Device
	bool cpu_only = true;

	torch::Device device( torch::kCPU );

	if( ! cpu_only ) {
		auto cuda_available = torch::cuda::is_available();
		device = cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
		std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
	} else {
		std::cout << "Training on CPU." << '\n';
	}

    auto pavgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    auto inp = torch::randn({1,3,7,7});
    auto outp = pavgpool->forward(inp);
    std::cout<<outp.sizes();

    std::vector<int> cfg_dd = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto vgg_dd = VGG(cfg_dd,1000,true);
    auto in = torch::randn({1,3,224,224});

    auto dictdd = vgg_dd->named_parameters();
    vgg_dd->forward(in);

    for (auto n = dictdd.begin(); n != dictdd.end(); n++) {
        std::cout<<(*n).key()<<std::endl;
    }

    std::string vgg_path = "./models/vgg16_bn.pt";
    std::string train_val_dir = "/media/stree/localssd/DL_data/hymenoptera_data";

    Classifier classifier(-1);

    if( ! cpu_only &&  torch::cuda::is_available() ) {
    	Classifier cls(0);
    	classifier = cls;
    }

    classifier.Initialize(2,vgg_path);

    classifier.Train(10, 8, 224, 0.0003, train_val_dir,".jpg", "./models/classifer.pt");

    //predict
    classifier.LoadWeight("./models/classifer.pt");
    cv::Mat image = cv::imread("./data/bee_img.jpeg");
    int ans = classifier.Predict(image);

    std::cout << (ans ? "predicted is bee." : "predicted is ant.") << '\n';
/*
    std::vector<int> cfg_a = {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    std::vector<int> cfg_d = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto vgg = VGG(cfg_d,1000,true);
    auto dict = vgg->named_parameters();
*/
    return 0;
}
