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
#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;

struct DNNModel : public torch::nn::Module {
	torch::Tensor w1, b1, w2, b2, w3, b3;
	DNNModel() {
        w1 = torch::randn({2,4});
        b1 = torch::zeros({1,4});
        w2 = torch::randn({4,8});
        b2 = torch::zeros({1,8});
        w3 = torch::randn({8,1});
        b3 = torch::zeros({1,1});

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
        y_pred = torch::where(y_pred > 0.5, torch::ones_like(y_pred, torch::kFloat32),
                          torch::zeros_like(y_pred, torch::kFloat32));
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
	auto r_p = 5.0 + torch::normal(0.0,1.0, {n_positive,1});
	auto theta_p = 2*M_PI*torch::rand({n_positive,1});
	auto Xp = torch::cat({r_p*torch::cos(theta_p),r_p*torch::sin(theta_p)}, /*axis =*/ 1);
	auto Yp = torch::ones_like(r_p);

	// 生成负样本, 大圆环分布
	auto r_n = 8.0 + torch::normal(0.0,1.0, {n_negative,1});
	auto theta_n = 2*M_PI*torch::rand({n_negative,1});
	auto Xn = torch::cat({r_n*torch::cos(theta_n),r_n*torch::sin(theta_n)}, /*axis =*/ 1);
	auto Yn = torch::zeros_like(r_n);

	//汇总样本
	auto X = torch::cat({Xp,Xn}, 0);
	auto Y = torch::cat({Yp,Yn}, 0);

	//可视化
	plt::figure_size(800,800);
	auto xp = Xp.index({Slice(),0});
	auto yp = Xp.index({Slice(),1});
	auto xn = Xn.index({Slice(),0});
	auto yn = Xn.index({Slice(),1});
	std::vector<float> xxp(xp.data_ptr<float>(), xp.data_ptr<float>() + xp.numel());
	std::vector<float> yyp(yp.data_ptr<float>(), yp.data_ptr<float>() + yp.numel());
	std::vector<float> xxn(xn.data_ptr<float>(), xn.data_ptr<float>() + xn.numel());
	std::vector<float> yyn(yn.data_ptr<float>(), yn.data_ptr<float>() + yn.numel());
	plt::scatter(xxp, yyp, 5.0, {{"c", "r"}, {"label", "positive"}});
	plt::scatter(xxn, yyn, 5.0, {{"c", "g"}, {"label", "negative"}});
	plt::legend();
	plt::show();

	// 测试数据管道效果
	int64_t batch_size = 8;
	auto dataloader = data_iter(X, Y, batch_size);
	torch::Tensor features = dataloader.front().first;
	torch::Tensor labels = dataloader.front().second;
	std::cout << features << std::endl;
	std::cout << labels << std::endl;

	// 测试模型结构
	auto model = DNNModel();
	model.to(device);

	torch::Tensor predictions = model.forward(features);

	torch::Tensor loss = model.loss_func(labels,predictions);
	torch::Tensor metric = model.metric_func(labels,predictions);

	std::cout << "init loss: " <<  loss.item<float>() << std::endl;
	std::cout << "init metric: " <<  metric.item<float>() << std::endl;

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
			torch::Tensor features = batch.first;
			torch::Tensor labels = batch.second;

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
			loss_list += loss.data().item<float>();
			metric_list += metric.data().item<float>();
			n_batch++;
		}

		if( (epoch+1) % 20 == 0 ) {
		    std::cout << "epoch = " << (epoch+1) << ", loss = " << loss_list/n_batch
		              << ", acc =" << metric_list/n_batch << std::endl;
		}
	}

	// # 结果可视化
	plt::figure_size(1200,500);
	plt::subplot(1,2,1);
	plt::scatter(xxp, yyp, 5.0, {{"c", "r"}, {"label", "positive"}});
	plt::scatter(xxn, yyn, 5.0, {{"c", "g"}, {"label", "negative"}});
	plt::legend();
	plt::title("y_true");

	plt::subplot(1,2,2);
	auto idx = torch::where(model.forward(X) >= 0.5);
	//std::cout << torch::index_select(X, /*dim =*/ 0, /*index =*/ idx[0]) << std::endl;
	auto Xp_pred = torch::index_select(X, /*dim =*/ 0, /*index =*/ idx[0]);
	idx = torch::where(model.forward(X) < 0.5);
	auto Xn_pred = torch::index_select(X, /*dim =*/ 0, /*index =*/ idx[0]);

	xp = Xp_pred.index({Slice(),0});
	yp = Xp_pred.index({Slice(),1});
    xn = Xn_pred.index({Slice(),0});
    yn = Xn_pred.index({Slice(),1});
	std::vector<float> xpp(xp.data_ptr<float>(), xp.data_ptr<float>() + xp.numel());
	std::vector<float> ypp(yp.data_ptr<float>(), yp.data_ptr<float>() + yp.numel());
	std::vector<float> xpn(xn.data_ptr<float>(), xn.data_ptr<float>() + xn.numel());
	std::vector<float> ypn(yn.data_ptr<float>(), yn.data_ptr<float>() + yn.numel());
	plt::scatter(xpp, ypp, 5.0, {{"c", "r"}, {"label", "positive"}});
	plt::scatter(xpn, ypn, 5.0, {{"c", "g"}, {"label", "negative"}});
	plt::title("y_pred");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}



