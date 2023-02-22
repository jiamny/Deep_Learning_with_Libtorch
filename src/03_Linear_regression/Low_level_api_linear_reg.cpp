#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <list>
#include <algorithm>
#include <unistd.h>
#include <iomanip>
#include <matplot/matplot.h>
using namespace matplot;

#include "../LRdataset.h"

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
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// #样本数量
	int64_t n = 400;

	// 生成测试用数据集
//	auto X = 10*torch::rand({n,2}) - 5.0;  									//torch.rand是均匀分布
	auto X = 10*torch::rand({n,1}).to(torch::kDouble) - 5.0;
	std::cout << "X:\n" << X.index({Slice(0, 10), Slice()}) << std::endl;
	auto w0 = torch::tensor({{2.0}}).to(torch::kDouble); //, {-3.0}});
	std::cout << "w0:\n" << w0.sizes() << std::endl;
	auto b0 = torch::tensor({{10.0}}).to(torch::kDouble);
	//auto Y = X.mm(w0) + b0 + torch::normal( 0.0, 2.0, /*size =*/ {n,1});  	// @表示矩阵乘法(Simon H operator),增加正态扰动
	auto Y = X*w0 + b0 + torch::normal( 0.0, 2.0, /*size =*/ {n,1}).to(torch::kDouble);
	std::cout << "Y:\n" << Y.sizes() << std::endl;

	// 数据可视化
	auto x1 = X.data().index({Slice(), 0});
//	std::cout << "x1:\n" << x1 << std::endl;
	auto y1 = Y.index({Slice(), 0});
	std::vector<double> xx(x1.data_ptr<double>(), x1.data_ptr<double>() + x1.numel());
	std::vector<double> yy(y1.data_ptr<double>(), y1.data_ptr<double>() + y1.numel());
	for( auto& n : xx )
		std::cout << n << ' ';
	std::cout << '\n';

	auto model = LinearRegression(w0, b0);
	model.to(device);
	model.train(true);

	int64_t epochs = 2000;
	int64_t batch_size = 10;
	torch::Tensor loss;
	std::cout << "111" << '\n';
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

	torch::Tensor yp;

	if( cuda_available )
		yp = model.w[0].cpu().data()*X.index({Slice(), 0})+model.b[0].cpu().data();
	else
		yp = model.w[0].data()*X.index({Slice(), 0})+model.b[0].data();

	std::vector<double> yyp(yp.data_ptr<double>(), yp.data_ptr<double>() + yp.numel());

	std::cout << model.w.sizes() << '\n';

    tiledlayout(1, 1);
    auto ax1 = nexttile();
    auto l = scatter(ax1, xx, yy, 6);
    l->marker_color({0.f, .5f, .5f});
    l->marker_face_color({0.f, .7f, .7f});
    hold(ax1, true);
    plot(ax1, xx, yyp, "-r");
    hold(ax1, false);

    show();

	std::cout << "Done!\n";
	return 0;
}




