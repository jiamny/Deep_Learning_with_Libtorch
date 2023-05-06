#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <list>
#include <algorithm>
#include <unistd.h>
#include <iomanip>
#include <cmath>

#include "../LRdataset.h"
#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

struct DNNModel : public torch::nn::Module {
	torch::Tensor w1, b1, w2, b2, w3, b3;
	DNNModel() {
        w1 = torch::randn({2,4}).to(torch::kDouble);
        b1 = torch::zeros({1,4}).to(torch::kDouble);
        w2 = torch::randn({4,8}).to(torch::kDouble);
        b2 = torch::zeros({1,8}).to(torch::kDouble);
        w3 = torch::randn({8,1}).to(torch::kDouble);
        b3 = torch::zeros({1,1}).to(torch::kDouble);

        register_parameter("w1", w1);
        register_parameter("b1", b1);
        register_parameter("w2", w2);
        register_parameter("b2", b2);
        register_parameter("w3", w3);
        register_parameter("b3", b3);
	}
    // 正向传播
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(x.mm(w1) + b1);
        x = torch::relu(x.mm(w2) + b2);
        auto y = torch::sigmoid(x.mm(w3) + b3);
        return y;
    }

    // 损失函数(二元交叉熵)
    torch::Tensor loss_func(torch::Tensor y_pred, torch::Tensor y_true) {
        //将预测值限制在1e-7以上, 1- (1e-7)以下，避免log(0)错误
        float eps = 1e-7;
        y_pred = torch::clamp(y_pred, eps, 1.0-eps);
        auto bce = - y_true*torch::log(y_pred) - (1-y_true)*torch::log(1-y_pred);
        return torch::mean(bce);
    }

    // 评估指标(准确率)
    torch::Tensor metric_func(torch::Tensor y_pred, torch::Tensor y_true) {
        y_pred = torch::where(y_pred > 0.5, torch::ones_like(y_pred, torch::kDouble),
                          torch::zeros_like(y_pred, torch::kDouble));
        auto acc = torch::mean(1-torch::abs(y_true-y_pred));
        return acc;
    }
};

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// 正负样本数量
	int64_t n_positive = 2000, n_negative = 2000;

	// 生成正样本, 小圆环分布
	auto r_p = 5.0 + torch::normal(0.0,1.0, {n_positive,1}).to(torch::kDouble);
	auto theta_p = 2*M_PI*torch::rand({n_positive,1}).to(torch::kDouble);
	auto Xp = torch::cat({r_p*torch::cos(theta_p),r_p*torch::sin(theta_p)}, /*axis =*/ 1).to(torch::kDouble);
	auto Yp = torch::ones_like(r_p).to(torch::kLong);

	// 生成负样本, 大圆环分布
	auto r_n = 8.0 + torch::normal(0.0,1.0, {n_negative,1}).to(torch::kDouble);
	auto theta_n = 2*M_PI*torch::rand({n_negative,1}).to(torch::kDouble);
	auto Xn = torch::cat({r_n*torch::cos(theta_n),r_n*torch::sin(theta_n)}, /*axis =*/ 1).to(torch::kDouble);
	auto Yn = torch::zeros_like(r_n).to(torch::kLong);

	//汇总样本
	auto X = torch::cat({Xp,Xn}, 0).to(torch::kDouble);
	auto Y = torch::cat({Yp,Yn}, 0).to(torch::kLong);
	std::cout << "Y:\n" << Y.sizes() << "\n";

	//可视化
	auto xp = Xp.index({Slice(),0});
	auto yp = Xp.index({Slice(),1});
	auto xn = Xn.index({Slice(),0});
	auto yn = Xn.index({Slice(),1});
	std::vector<double> xxp(xp.data_ptr<double>(), xp.data_ptr<double>() + xp.numel());
	std::vector<double> yyp(yp.data_ptr<double>(), yp.data_ptr<double>() + yp.numel());
	std::vector<double> xxn(xn.data_ptr<double>(), xn.data_ptr<double>() + xn.numel());
	std::vector<double> yyn(yn.data_ptr<double>(), yn.data_ptr<double>() + yn.numel());

	// 测试数据管道效果
	int64_t batch_size = 8;
	auto dataloader = data_iter(X, Y, batch_size);
	torch::Tensor features = dataloader.front().first.to(device);
	torch::Tensor labels = dataloader.front().second.to(device);
	std::cout << "features:\n" << features << std::endl;
	std::cout << "labels:\n" << labels << std::endl;

	// 测试模型结构
	auto model = DNNModel();
	model.to(device);

	torch::Tensor predictions = model.forward(features);

	torch::Tensor loss = model.loss_func(predictions, labels);
	torch::Tensor metric = model.metric_func(predictions, labels);

	std::cout << "init loss: " <<  loss.cpu().item<double>() << std::endl;
	std::cout << "init metric: " <<  metric.cpu().item<double>() << std::endl;

	int64_t epochs = 1000;
	batch_size = 20;

	float loss_list, metric_list;
	// 测试train_step效果
	for( int64_t epoch = 0; epoch < epochs; epoch++ ) {
		model.train(true);

		auto dataloader =  data_iter(X, Y, batch_size);
		loss_list = 0.0;
		metric_list = 0.0;
		int64_t n_batch = 0;
		for( auto& batch : dataloader) {
			torch::Tensor features = batch.first.to(device);
			torch::Tensor labels = batch.second.to(device);

			//正向传播求损失
			predictions = model.forward(features);
			loss = model.loss_func(predictions,labels);
    		metric = model.metric_func(predictions,labels);

    		// 反向传播求梯度
    		loss.backward();

    		torch::NoGradGuard no_grad;
    		// 梯度下降法更新参数
    		for( auto& param : model.parameters() ) {
        		//注意是对param.data进行重新赋值,避免此处操作引起梯度记录
        		param.data() = (param.data() - 0.01*param.grad().data());
    		}

    		// 梯度清零
			model.zero_grad();
			loss_list += loss.cpu().data().item<float>();
			metric_list += metric.cpu().data().item<float>();
			n_batch++;
		}

		if( (epoch+1) % 20 == 0 ) {
		    std::cout << "epoch = " << (epoch+1) << ", loss = " << loss_list/n_batch
		              << ", acc = " << metric_list/n_batch << std::endl;
		}
	}

	auto idx = torch::where(model.forward(X.to(device)) >= 0.5);
	//std::cout << torch::index_select(X, /*dim =*/ 0, /*index =*/ idx[0]) << std::endl;
	auto Xp_pred = torch::index_select(X.to(device), /*dim =*/ 0, /*index =*/ idx[0]);
	idx = torch::where(model.forward(X.to(device)) < 0.5);
	auto Xn_pred = torch::index_select(X.to(device), /*dim =*/ 0, /*index =*/ idx[0]);

	xp = Xp_pred.index({Slice(),0}).cpu();
	yp = Xp_pred.index({Slice(),1}).cpu();
    xn = Xn_pred.index({Slice(),0}).cpu();
    yn = Xn_pred.index({Slice(),1}).cpu();

	std::vector<double> xpp(xp.data_ptr<double>(), xp.data_ptr<double>() + xp.numel());
	std::vector<double> ypp(yp.data_ptr<double>(), yp.data_ptr<double>() + yp.numel());
	std::vector<double> xpn(xn.data_ptr<double>(), xn.data_ptr<double>() + xn.numel());
	std::vector<double> ypn(yn.data_ptr<double>(), yn.data_ptr<double>() + yn.numel());

	// # 结果可视化
	auto h = figure(true);
	h->size(500, 1000);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(2, 1);
	h->position(0, 0);

    auto ax1 = h->nexttile();
    auto l = scatter(ax1, xxp, yyp, 5);
    l->marker_color({0.f, .5f, .5f});
    l->marker_face_color({0.f, .7f, .7f});
	hold(ax1, true);
	auto l2 = scatter(ax1, xxn, yyn, 7);
	l2->marker_color({0.2f, .25f, .5f});
	l2->marker_face_color({0.3f, .35f, .7f});
	legend(ax1, "positive", "negative");
	title(ax1, "y true");

    auto ax2 = h->nexttile();
    auto l3 = scatter(ax2, xpp, ypp, 5);
    l3->marker_color({0.f, .5f, .5f});
    l3->marker_face_color({0.f, .7f, .7f});
	hold(ax2, true);
	auto l4 = scatter(ax2, xpn, ypn, 7);
	l4->marker_color({0.2f, .25f, .5f});
	l4->marker_face_color({0.3f, .35f, .7f});
	legend(ax2, "positive", "negative");
	title(ax2, "y pred");
	show();

	std::cout << "Done!\n";
	return 0;
}



