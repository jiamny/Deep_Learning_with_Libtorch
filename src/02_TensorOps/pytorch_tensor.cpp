#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <map>

using namespace torch::autograd;

int main() {

	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

	/***************************************************************
	 * Create a torch::Tensor from C/C++ array
	 ***************************************************************/
	float array[] = { 1, 2, 3, 4, 5};

	torch::Tensor tharray = torch::from_blob(array, {5}, options);

	std::cout << "Tensor from c/c++ array = \n" <<  tharray << '\n';

	TORCH_CHECK(array == tharray.data_ptr<float>());

	// Tensor from vector:
	std::vector<double> data_vector = {1, 2, 3, 4};
	torch::Tensor t2 = torch::from_blob(data_vector.data(), {2, 2}, dtype(torch::kDouble));
	std::cout << "Tensor from vector:\n" << t2 << "\n";

	TORCH_CHECK(data_vector.data() == t2.data_ptr<double>());

	// Change tensor contents
	auto points = torch::zeros(6);
	std::cout << "Points =\n" << points << "\n";
	points[0] = 4.0;
	points[1] = 1.0;
	points[2] = 5.0;
	points[3] = 3.0;
	points[4] = 2.0;
	points[5] = 1.0;
	std::cout << "Changed contents points =\n" << points << "\n";

//	float farray[] = {4.0, 1.0, 5.0, 3.0, 2.0, 1.0};
//	points = torch::from_blob(farray, {3,2}, options);

	points = torch::tensor({{4.0, 1.0}, {5.0, 3.0}, {2.0, 1.0}}, options);
	std::cout << "Points =\n" << points << "\n";
	std::cout << "Points storage = " << points.storage() << "\n";
	std::cout << "Points size = " << points.sizes() << "\n";

	auto points_storage = points.storage().data();
	std::cout << "---- points = \n" << points_storage << "\n";

	std::cout << "points[0][0] = " << points[0][0] << "\n";
	std::cout << "points[0][1] = " << points[0][1] << "\n";

	points[0][0] = 2.0;
	std::cout << "Points =\n" << points << "\n";

	auto second_point = points[0][1];
	std::cout << "second_point.size = " << second_point.sizes() << '\n';
	std::cout << "second_point.storage_offset() = " << second_point.storage_offset() << '\n';

	std::cout << "second_point.stride() = " << second_point.strides() << '\n';

	std::vector<float> td;
	td.push_back(11.0);
	td.push_back(12.0);
	td.push_back(13.0);
	td.push_back(21.0);
	td.push_back(22.0);
	td.push_back(23.0);
	torch::Tensor tt = torch::from_blob(td.data(), {2, 3}).clone();;
	std::cout << "TT = " << tt << "\n";

	/***************************************************************
	 * Tensor transpose
	***************************************************************/
	auto points_t = points.t();
	std::cout << "points_t =\n" << points_t << "\n";

	TORCH_CHECK(points.storage().data_ptr() == points_t.storage().data_ptr())

	std::cout << "points.stride() = " << points.strides() << '\n';
	std::cout << "points_t.stride() = " << points_t.strides() << '\n';

	auto some_t = torch::ones({3, 4, 5});
	auto transpose_t = some_t.transpose(0, 2);

	std::cout << "some_t.shape = " << some_t.sizes() << '\n';
	std::cout << "transpose_t.shape = " << transpose_t.sizes() << '\n';
	std::cout << "some_t.stride() = " << some_t.strides() << '\n';
	std::cout << "transpose_t.stride() = " << transpose_t.strides() << '\n';

	std::cout << "some_t.is_contiguous() = " << some_t.is_contiguous() << '\n';
	std::cout << "transpose_t.is_contiguous() = " << transpose_t.is_contiguous() << '\n';

	auto points_t_cont = transpose_t.contiguous();

	std::cout << "points_t_cont.shape = " << points_t_cont.sizes() << '\n';
	std::cout << "points_t_cont.stride() = " << points_t_cont.strides() << '\n';
	std::cout << "points_t_cont.storage() = " << points_t_cont.storage() << '\n';

	auto double_points = torch::ones({10, 2}, options=torch::kDouble);
	auto short_points = torch::tensor({{1, 2}, {3, 4}}, options=torch::kShort);

	std::cout << "double_points = \n" << double_points << '\n';
	std::cout << "short_points = \n" << short_points << '\n';

	std::cout << "double_points.dtype = " << double_points.dtype() << '\n';
	std::cout << "short_points.dtype = " << short_points.dtype() << '\n';

	auto float_points = torch::zeros({10, 2}).to(torch::kFloat);
	auto long_points = torch::ones({10, 2}).to(torch::kLong);

	std::cout << "float_points.dtype = " << float_points.dtype() << '\n';
	std::cout << "long_points.dtype = " << long_points.dtype() << '\n';

	//	torch::save(double_points, "./src/01_Introducing/pytorch_basics/output/ourpoints.t");

	/******************************************************************
	 * SLICING AND EXTRACTING PARTS FROM TENSORS
	 ******************************************************************/
	using torch::indexing::Slice;
	using torch::indexing::None;
	using torch::indexing::Ellipsis;

	auto x = torch::tensor({{1, 2, 3, 4}, {4, 5, 6, 7}, {1, 2, 3, 4}, {7, 8, 9, 10}}, options);
	std::cout <<  " x[0] = " << x[0] << '\n';

	std::cout <<  " x[0,2] = "     << x.index({0, 2}) << '\n';
	std::cout <<  " x[,2] = \n"    << x.index({Slice(), 2}) << '\n';
	std::cout <<  " x[1,0:2] = \n" << x.index({1, Slice(None, 2)}) << '\n';

	std::cout << "x.shape = "     << x.sizes() << '\n';
	std::cout << "x.view(-1) = "  << x.view(-1) << '\n';
	std::cout << "x.view(4,4) = " << x.view({4,4}) << '\n';
	std::cout << "x.view(8,2) = " << x.view({8,2}) << '\n';

	std::cout << "\"x[:,1:]\":\n"    << x.index({Slice(), Slice(1, None)}) << '\n';
	std::cout << "\"x[:,::2]\":\n"   << x.index({Slice(), Slice(None, None, 2)}) << '\n';
	std::cout << "\"x[:2,1]\":\n"    << x.index({Slice(None, 2), 1}) << '\n';
	std::cout << "\"x[..., :2]\":\n" << x.index({Ellipsis, Slice(None, 2)}) << "\n\n";

	/******************************************************************
	 * Stack tensors
	 ******************************************************************/
	auto grad1 = torch::empty(3);
	auto grad2 = torch::empty(3);
	auto grad3 = torch::ones(3);
	auto t_lst = torch::stack({grad1, grad2, grad3}).to(options).view({1,9}); // default is  3 x 3

	std::cout << "stack tensor = \n" <<  t_lst << '\n';

	auto f_t = torch::stack({grad1, grad2, grad3}).to(torch::kFloat32).view({3,3});
	std::cout << "stack tensor = \n" << f_t << '\n';

	/***************************************************************
	 * Tensor operations
	 ***************************************************************/
	std::cout << "swaps all axis at once\n";

	auto a = torch::ones({1,2,3,4});
	std::cout << "a = \n" << a.sizes() << '\n';
	std::cout << "a.permute(3,2,1,0).size()) = \n" << a.permute({3,2,1,0}).sizes() << '\n';  //swaps all axis at once

	// A tensor can be split between multiple chunks. Those small chunks
	// can be created along dim rows and dim columns. The following example
	// shows a sample tensor of size (4,4). The chunk is created using the third
	// argument in the function, as 0 or 1.

	std::cout << "to split a tensor among small chunks\n";
	a = torch::randn({4, 4});
	std::cout << "a = \n" << a << '\n';
	auto b = torch::chunk(a, 2);
	std::cout << "b = \n" << b << '\n';

	b = torch::chunk(a, 2, 0);
	std::cout << "b = \n" << b << '\n';

	b = torch::chunk(a, 2, 1);
	std::cout << "b = \n" << b << '\n';

	// The gather function collects elements from a tensor and places it in
	// another tensor using an index argument. The index position is determined
	// by the LongTensor function in PyTorch
	a = torch::tensor({{11,12},{23,24}});
	std::cout << "a = \n" << a << '\n';

//	b = torch::gather(a, 1, torch::tensor({{0,0}, {1,0}}));
//	std::cout << "b = \n" << b << '\n';

	// The LongTensor function or the index select function can be used to
	// fetch relevant values from a tensor. The following sample code shows two
	// options: selection along rows and selection along columns. If the second
	// argument is 0, it is for rows. If it is 1, then it is along the columns.

	a = torch::randn({4, 4});
	std::cout << "a = \n" << a << '\n';
	auto indices = torch::tensor({0, 1});

	std::cout << "Torch.index_select(a, 0, indices)=\n" << torch::index_select(a, 0, indices) << '\n';

	// It is a common practice to check non-missing values in a tensor, the
	// objective is to identify non-zero elements in a large tensor.
	// identify null input tensors using nonzero function
	std::cout << torch::nonzero(torch::tensor({10.0,0.0,23.0,0.0,0.0})) << '\n';

	std::cout << "Splitting the tensor into small chunks:\n";
	std::cout << torch::split(torch::tensor({12,21,34,32,45,54,56,65}),2) << '\n';

	// The split function splits a long tensor into smaller tensors.
	// splitting the tensor into small chunks
	std::cout << torch::split(torch::tensor({12,21,34,32,45,54,56,65}),3) << '\n';

	// how to remove a dimension from a tensor
	// The unbind function removes a dimension from a tensor. To remove
	// the dimension row, the 0 value needs to be passed. To remove a column,
	// the 1 value needs to be passed.
	std::cout << "Remove column dimension from a tensor:\n";
	x = torch::randn({4,5});

	std::cout << "x = \n" << x << '\n';
	std::cout << torch::unbind(x,1) << '\n'; // dim=1 removing column dimension

	std::cout << "Remove row dimension from a tensor:\n";
	std::cout << torch::unbind(x,0) << '\n'; // dim=0 removing row dimension

	std::cout << "adding value to the existing tensor, scalar addition:\n";
	std::cout << torch::add(x,20) << '\n';

	std::cout << "scalar multiplication:\n";
	std::cout << torch::mul(x,2) << '\n';

	std::cout << "round up tensor values:\n";
	torch::manual_seed(1234);
	std::cout << torch::ceil(torch::randn({5,5})) << '\n';

	std::cout << "flooring down tensor values:\n";
	torch::manual_seed(1234);
	std::cout << torch::floor(torch::randn({5,5})) << '\n';

	// Limiting the values of any tensor within a certain range can be done
	// using the minimum and maximum argument and using the clamp function
	// truncate the values in a range say 0,1
	torch::manual_seed(1234);


	//std::cout << torch::clamp_max(clamp_max(torch::randn({5,5}), 0.5) << '\n';

	std::cout << "scalar division:\n";
	std::cout << torch::div(x,0.2) << '\n';

	std::cout << "the exponential of a tensor:\n";
	std::cout << torch::exp(x) << '\n';

	std::cout << "get the fractional portion of each tensor:\n";
	std::cout << "x = \n" << torch::add(x,100) << '\n';
	std::cout << torch::frac(torch::add(x,100)) << '\n';

	std::cout << "compute the log of the values in a tensor:\n";
	std::cout << torch::log(x) << '\n';

	std::cout << "round the values in a tensor:\n";
	std::cout << torch::round(x) << '\n';

	std::cout << "compute the sigmoid of the input tensor:\n";
	std::cout << torch::sigmoid(x) << '\n';

	std::cout << "finding the square root of the values:\n";
	std::cout << torch::sqrt(torch::abs(x)) << '\n';

	x = torch::randn({3,3});
	auto y = torch::randn({3,3});
	auto z = torch::add(x,y).clone();
	std::cout << "x = " << x << std::endl;
	std::cout << "y = " << y << std::endl;
	std::cout << "z = " << z << std::endl;

	std::map<std::string, std::vector<int64_t>>  cfg;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("in_planes", {96,192,384,768}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("out_planes", {256,512,1024,2048}));
	std::vector<int64_t> o_planes = cfg.at("out_planes");
	std::cout << o_planes[1] << std::endl;

	std::cout << "Done!\n";
	return(0);
}


