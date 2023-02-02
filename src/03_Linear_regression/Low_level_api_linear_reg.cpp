#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <list>
#include <algorithm>
#include <unistd.h>
#include <iomanip>

#include "../LRdataset.h"
#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;

// 定义模型
struct LinearRegression : public torch::nn::Module {
	torch::Tensor w;
	torch::Tensor b;
	LinearRegression() {}
	LinearRegression(torch::Tensor w0, torch::Tensor b0){
        w = torch::randn_like(w0, torch::requires_grad(true));
        b = torch::zeros_like(b0, torch::requires_grad(true));
        register_parameter("w", w);
        register_parameter("b", b);
	}

    //正向传播
    torch::Tensor forward(torch::Tensor x) {
        return x.mm(w) + b;
    }

    // 损失函数
    torch::Tensor  loss_func(torch::Tensor y_pred, torch::Tensor y_true) {
        return torch::mean(torch::pow((y_pred - y_true), 2) / 2.0);
    }
};


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();

	torch::Device device = torch::Device(torch::kCPU);

	if( cuda_available ) {
		int gpu_id = 0;
		device = torch::Device(torch::kCUDA, gpu_id);

		if(gpu_id >= 0) {
			if(gpu_id >= torch::getNumGPUs()) {
				std::cout << "No GPU id " << gpu_id << " abailable, use CPU." << std::endl;
				device = torch::Device(torch::kCPU);
				cuda_available = false;
			} else {
				device = torch::Device(torch::kCUDA, gpu_id);
			}
		} else {
			device = torch::Device(torch::kCPU);
			cuda_available = false;
		}
	}

	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::cout << device << '\n';

	torch::manual_seed(1000);

	// #样本数量
	int64_t n = 400;

	// 生成测试用数据集
//	auto X = 10*torch::rand({n,2}) - 5.0;  									//torch.rand是均匀分布
	auto X = 10*torch::rand({n,1}) - 5.0;
	std::cout << "X:\n" << X.index({Slice(0, 10), Slice()}) << std::endl;
	auto w0 = torch::tensor({{2.0}}); //, {-3.0}});
	std::cout << "w0:\n" << w0.sizes() << std::endl;
	auto b0 = torch::tensor({{10.0}});
	//auto Y = X.mm(w0) + b0 + torch::normal( 0.0, 2.0, /*size =*/ {n,1});  	// @表示矩阵乘法(Simon H operator),增加正态扰动
	auto Y = X*w0 + b0 + torch::normal( 0.0, 2.0, /*size =*/ {n,1});
	std::cout << "Y:\n" << Y.sizes() << std::endl;

	// 数据可视化
	plt::figure_size(800, 600);
//	plt::subplot2grid(1, 2, 0, 0, 1, 1);
	auto x1 = X.data().index({Slice(), 0});
//	std::cout << "x1:\n" << x1 << std::endl;
	auto y1 = Y.index({Slice(), 0});
	std::vector<float> xx(x1.data_ptr<float>(), x1.data_ptr<float>() + x1.numel());
	std::vector<float> yy(y1.data_ptr<float>(), y1.data_ptr<float>() + y1.numel());
	plt::scatter(xx, yy, 5.0, {{"c", "b"}, {"label", "samples"}});
	plt::legend();
	plt::xlabel("x1");
	plt::ylabel("y");
	plt::show();
	plt::close();
/*
	plt::subplot2grid(1, 2, 0, 1, 1, 1);
	auto x2 = X.data().index({Slice(), 1});
	std::vector<float> xx2(x2.data_ptr<float>(), x2.data_ptr<float>() + x2.numel());
	plt::scatter(xx2, yy, 5.0, {{"c", "r"}, {"label", "samples"}});
	plt::legend();
	plt::xlabel("x2");
	plt::ylabel("y");
	plt::show();
	plt::close();
*/
	auto model = LinearRegression(w0, b0);
	model.to(device);
	model.train(true);

	int64_t epochs = 2000;
	int64_t batch_size = 10;
	torch::Tensor loss;

	// 测试train_step效果
	for( int64_t epoch = 0; epoch < epochs; epoch++ ) {
		model.train(true);

		auto dataloader =  data_iter(X, Y, batch_size);

		for( auto& batch : dataloader) {
			auto features = batch.first.to(device);
			auto labels = batch.second.to(device);
			//std::cout << "x:\n" << features << "\n";
			//std::cout << "y:\n" << labels << "\n";
			auto predictions = model.forward(features);
			//std::cout << "predictions:\n" << predictions << "\n";
			loss = model.loss_func(predictions,labels);
			//std::cout << "loss:\n" << loss << "\n";

			//反向传播求梯度
			loss.backward();

			torch::NoGradGuard no_grad;

			//梯度下降法更新参数
			// variable += any_thing is inplace but variable = variable + any_thing is NOT inplace
			model.w -= 0.001*model.w.grad();
			model.b -= 0.001*model.b.grad();

			// 梯度清零
			model.w.grad().zero_();
			model.b.grad().zero_();
		}

		if( (epoch+1) % 200 == 0 ) {
		    std::cout << "epoch = " << (epoch+1) << ", loss = " << loss.item<float>()
		              << ", model.w = " << model.w.data() << ", model.b =" << model.b.data() << std::endl;
		}
	}

	plt::figure_size(800, 600);
//  plt::subplot2grid(1, 2, 0, 0, 1, 1);
	plt::scatter(xx, yy, 5.0, {{"c", "g"}, {"label", "samples"}});
	torch::Tensor yp;

	if(cuda_available )
		yp = model.w[0].cpu().data()*X.index({Slice(), 0})+model.b[0].cpu().data();
	else
		yp = model.w[0].data()*X.index({Slice(), 0})+model.b[0].data();

	std::vector<float> yyp(yp.data_ptr<float>(), yp.data_ptr<float>() + yp.numel());
	plt::named_plot("model", xx, yyp, "-r");
    plt::legend();
    plt::xlabel("x1");
    plt::ylabel("y");
    plt::show();
  	plt::close();
/*
    plt::subplot2grid(1, 2, 0, 1, 1, 1);
    plt::scatter(xx2, yy, 2.0, {{"c", "g"}, {"label", "samples"}});
    yp = model.w[1].data()*X.index({Slice(), 1})+model.b[0].data();
    std::vector<float> yyp2(yp.data_ptr<float>(), yp.data_ptr<float>() + yp.numel());
//    plt::plot(xx, yyp2, "-r"); //, 5.0, "model");
    plt::scatter(xx2, yyp2, 5.0, {{"c", "r"}, {"label", "model"}});
//    plt::legend();
    plt::xlabel("x2");
    plt::ylabel("y");
    plt::show();
	plt::close();
*/
	std::cout << "Done!\n";
	return 0;
}




