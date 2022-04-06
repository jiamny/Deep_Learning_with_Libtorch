#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
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

// Focal loss
class FocalLoss : public torch::nn::Module {
public:
	FocalLoss(float gamma=2.0, float alpha=0.75) {
        this->gamma = gamma;
        this->alpha = alpha;
	}

    torch::Tensor forward(torch::Tensor y_pred, torch::Tensor y_true) {
        auto bce = torch::nn::BCELoss(torch::nn::BCELossOptions().reduction(torch::kNone))(y_pred,y_true); // reduction = "none"
        auto p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred));
        auto alpha_factor = y_true * this->alpha + (1 - y_true) * (1 - this->alpha);
        auto modulating_factor = torch::pow(1.0 - p_t, this->gamma);
        auto loss = torch::mean(alpha_factor * modulating_factor * bce);
        return loss;
    }
private:
    float gamma, alpha;
};

// define DNN model
struct DNNModel : public torch::nn::Module  {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	DNNModel() {
        fc1 = torch::nn::Linear(2,4);
        fc2 = torch::nn::Linear(4,8);
        fc3 = torch::nn::Linear(8,1);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
	}

    torch::Tensor forward( torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto y = torch::nn::Sigmoid()(fc3->forward(x));
        return y;
    }
};

// 准确率
torch::Tensor accuracy(torch::Tensor y_pred, torch::Tensor y_true) {
    y_pred = torch::where(y_pred > 0.5, torch::ones_like(y_pred).to(torch::kFloat),
                      torch::zeros_like(y_pred).to( torch::kFloat));
    auto acc = torch::mean(1-torch::abs(y_true-y_pred));
    return acc;
}

// L2正则化
template<typename T>
torch::Tensor L2Loss(T& model, float alpha) {
    auto l2_loss = torch::tensor(0.0, torch::requires_grad(true));

    for( auto& param : model.named_parameters()) {
    	auto pair = param.pair(); // (key, value)
    	if( pair.first.find("bias") == std::string::npos ) { //一般不对偏置项使用正则
    		//std::cout << pair.second << "\n";
    		l2_loss = l2_loss + (0.5 * alpha * torch::sum(torch::pow(pair.second, 2)));
    	}
    }
    return l2_loss;
}

// L1正则化
template<typename T>
torch::Tensor L1Loss(T& model, float beta) {
    auto l1_loss = torch::tensor(0.0, torch::requires_grad(true));
    //for name, param in model.named_parameters():
    //    if 'bias' not in name:
    //        l1_loss = l1_loss +  beta * torch.sum(torch.abs(param))
    for( auto& param : model.named_parameters()) {
        auto pair = param.pair(); // (key, value)
        if( pair.first.find("bias") == std::string::npos ) {
        	l1_loss = l1_loss +  beta * torch::sum(torch::abs(pair.second));
        }
    }
    return l1_loss;
}

// 将L2正则和L1正则添加到FocalLoss损失，一起作为目标函数
template<typename T>
torch::Tensor focal_loss_with_regularization(T& model, torch::Tensor y_pred, torch::Tensor y_true) {
    auto fl = FocalLoss();
    auto focal = fl.forward(y_pred,y_true);
    auto l2_loss = L2Loss(model,0.001); 	//注意设置正则化项系数
    auto l1_loss = L1Loss(model,0.001);
    auto total_loss = focal + l2_loss + l1_loss;
    return total_loss;
}

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	/*************************************************************************************
	 一般来说，监督学习的目标函数由损失函数和正则化项组成。(Objective = Loss + Regularization)

	Pytorch中的损失函数一般在训练模型时候指定。

	注意Pytorch中内置的损失函数的参数和tensorflow不同，是y_pred在前，y_true在后，而Tensorflow是y_true在前，y_pred在后。
	对于回归模型，通常使用的内置损失函数是均方损失函数nn.MSELoss 。

	对于二分类模型，通常使用的是二元交叉熵损失函数nn.BCELoss (输入已经是sigmoid激活函数之后的结果)
	或者 nn.BCEWithLogitsLoss (输入尚未经过nn.Sigmoid激活函数) 。

	对于多分类模型，一般推荐使用交叉熵损失函数 nn.CrossEntropyLoss。
	(y_true需要是一维的，是类别编码。y_pred未经过nn.Softmax激活。)

	此外，如果多分类的y_pred经过了nn.LogSoftmax激活，可以使用nn.NLLLoss损失函数(The negative log likelihood loss)。
	这种方法和直接使用nn.CrossEntropyLoss等价。

	如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量y_pred，y_true作为输入参数，并输出一个标量作为损失函数值。
	Pytorch中的正则化项一般通过自定义的方式和损失函数一起添加作为目标函数。
	如果仅仅使用L2正则化，也可以利用优化器的weight_decay参数来实现相同的效果。
	 */

	// --------------------------------------------
	// 一，内置损失函数
	// --------------------------------------------
	auto y_pred = torch::tensor({{10.0,0.0,-10.0},{8.0,8.0,8.0}});
	auto y_true = torch::tensor({0,2});

	// 直接调用交叉熵损失
	auto ce = torch::nn::CrossEntropyLoss()(y_pred,y_true);
	std::cout << "ce: " << ce << "\n";

	// 等价于先计算nn.LogSoftmax激活，再调用NLLLoss
	auto y_pred_logsoftmax = torch::nn::LogSoftmax(/*dim =*/ 1)(y_pred);
	auto nll = torch::nn::NLLLoss()(y_pred_logsoftmax,y_true);
	std::cout << "nll: " << nll << "\n";

	/**************************************************************************************
	内置的损失函数一般有类的实现和函数的实现两种形式。
	如：nn.BCE 和 F.binary_cross_entropy 都是二元交叉熵损失函数，前者是类的实现形式，后者是函数的实现形式。

	实际上类的实现形式通常是调用函数的实现形式并用nn.Module封装后得到的。
	一般我们常用的是类的实现形式。它们封装在torch.nn模块下，并且类名以Loss结尾。
	常用的一些内置损失函数说明如下。
	 * nn.MSELoss（均方误差损失，也叫做L2损失，用于回归）
	 * nn.L1Loss （L1损失，也叫做绝对值误差损失，用于回归）
	 * nn.SmoothL1Loss (平滑L1损失，当输入在-1到1之间时，平滑为L2损失，用于回归)
	 * nn.BCELoss (二元交叉熵，用于二分类，输入已经过nn.Sigmoid激活，对不平衡数据集可以用weigths参数调整类别权重)
	 * nn.BCEWithLogitsLoss (二元交叉熵，用于二分类，输入未经过nn.Sigmoid激活)
	 * nn.CrossEntropyLoss (交叉熵，用于多分类，要求label为稀疏编码，输入未经过nn.Softmax激活，对不平衡数据集可以用weigths参数调整类别权重)
	 * nn.NLLLoss (负对数似然损失，用于多分类，要求label为稀疏编码，输入经过nn.LogSoftmax激活)
	 * nn.CosineSimilarity(余弦相似度，可用于多分类)
	 	 * nn.AdaptiveLogSoftmaxWithLoss (一种适合非常多类别且类别分布很不均衡的损失函数，会自适应地将多个小类别合成一个cluster)
	 */

	// --------------------------------
	// 二，自定义损失函数
	// --------------------------------
	/****************************************************************************
	自定义损失函数接收两个张量y_pred,y_true作为输入参数，并输出一个标量作为损失函数值。
	也可以对nn.Module进行子类化，重写forward方法实现损失的计算逻辑，从而得到损失函数的类的实现。

	下面是一个Focal Loss的自定义实现示范。Focal Loss是一种对binary_crossentropy的改进损失函数形式。
	它在样本不均衡和存在较多易分类的样本时相比binary_crossentropy具有明显的优势。
	它有两个可调参数，alpha参数和gamma参数。其中alpha参数主要用于衰减负样本的权重，gamma参数主要用于衰减容易训练样本的权重。
	从而让模型更加聚焦在正样本和困难样本上。这就是为什么这个损失函数叫做Focal Loss。
	 */
	// 困难样本
	auto y_pred_hard = torch::tensor({{0.5},{0.5}});
	auto y_true_hard = torch::tensor({{1.0},{0.0}});

	//容易样本
	auto y_pred_easy = torch::tensor({{0.9},{0.1}});
	auto y_true_easy = torch::tensor({{1.0},{0.0}});

	auto focal_loss = FocalLoss();
	auto bce_loss = torch::nn::BCELoss();

	std::cout << "focal_loss(hard samples): " << focal_loss.forward(y_pred_hard,y_true_hard) << "\n";
	std::cout << "bce_loss(hard samples): " << bce_loss->forward(y_pred_hard,y_true_hard) << "\n";
	std::cout << "focal_loss(easy samples): " << focal_loss.forward(y_pred_easy,y_true_easy) << "\n";
	std::cout << "bce_loss(easy samples): " << bce_loss->forward(y_pred_easy,y_true_easy) << "\n";

	// 可见 focal_loss让容易样本的权重衰减到原来的 0.0005/0.1054 = 0.00474
	// 而让困难样本的权重只衰减到原来的 0.0866/0.6931=0.12496
	// 因此相对而言，focal_loss可以衰减容易样本的权重。

	// --------------------------------------
	// 三，自定义L1和L2正则化项
	// --------------------------------------
	/*
	通常认为L1 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择。
	而L2 正则化可以防止模型过拟合（overfitting）。一定程度上，L1也可以防止过拟合。
	下面以一个二分类问题为例，演示给模型的目标函数添加自定义L1和L2正则化项的方法。
	这个范例同时演示了上一个部分的FocalLoss的使用。
	 */
	//正负样本数量
	int64_t n_positive = 200, n_negative = 6000;

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
	//std::cout << "Y:\n" << Y << "\n";

	bool show_plot = true;

	if( show_plot ) {
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
		plt::close();
	}

	auto model = DNNModel();
	model.to(device);
	std::cout << model << "\n";

	int64_t batch_size = 8;
	auto dataloader = data_iter(X, Y, batch_size);
	torch::Tensor features = dataloader.front().first;
	torch::Tensor labels = dataloader.front().second;
	std::cout << features << std::endl;
	std::cout << labels << std::endl;

	torch::Tensor predictions = model.forward(features);

	torch::Tensor loss = focal_loss_with_regularization(model, predictions, labels); //model.loss_func(labels,predictions);
	torch::Tensor metric = accuracy(predictions, labels);
	std::cout << "init loss: " <<  loss.item<float>() << std::endl;
	std::cout << "init metric: " <<  metric.item<float>() << std::endl;

	int64_t epochs = 500;
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
			loss = focal_loss_with_regularization(model, predictions, labels);
	    	metric = accuracy(predictions,labels);

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
			          << ", acc = " << metric_list/n_batch << std::endl;
		}
	}

	if( show_plot ) {
		// # 结果可视化
		plt::figure_size(1200,500);
		plt::subplot2grid(1, 2, 0, 0, 1, 1);
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
		plt::title("y_true");

		plt::subplot2grid(1, 2, 0, 1, 1, 1);
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
	}

	// -----------------------------------------
	// 四，通过优化器实现L2正则化
	//
	// 如果仅仅需要使用L2正则化，那么也可以利用优化器的weight_decay参数来实现。weight_decay参数可以设置参数在训练过程中的衰减，这和L2正则化的作用效果等价。
	// before L2 regularization:
	// gradient descent: w = w - lr * dloss_dw
	// after L2 regularization:
	// gradient descent: w = w - lr * (dloss_dw+beta*w) = (1-lr*beta)*w - lr*dloss_dw
	// so （1-lr*beta）is the weight decay ratio.
	//
	// Pytorch的优化器支持一种称之为Per-parameter options的操作，就是对每一个参数进行特定的学习率，权重衰减率指定，以满足更为细致的要求
	// -----------------------------------------
	//weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
	//bias_params = [param for name, param in model.named_parameters() if "bias" in name]

	//auto optimizer = torch::optim::SGD({{"params": weight_params, "weight_decay":1e-5},
	//                             {"params": bias_params, "weight_decay":0}},
	//                            /*lr=*/1e-2, /*momentum=*/0.9);

	auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(1e-2).weight_decay(1e-5).momentum(0.9));
	std::cout << "Done!\n";
	return 0;
}


