
#include <iostream>
#include <stdexcept>
#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <iomanip>

#include "../../matplotlibcpp.h"

namespace plt = matplotlibcpp;


using Data = std::vector<std::pair<std::string, long>>;
using Example = torch::data::Example<>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

 public:
  CustomDataset(const Data& dt, std::string fileDir, int imgSize) {
	  data=dt;
	  datasetPath=fileDir;
	  image_size=imgSize;
  }

  Example get(size_t index) {
    std::string path = datasetPath + data[index].first;
//    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
//    std::cout << "path = " << path << " label = " << tlabel << "\n";
//    auto mat = cv::imread(path.c_str(), 1);
    cv::Mat mat = cv::imread(path.c_str(), cv::IMREAD_COLOR);

    if(! mat.empty() ) {
//    	cv::namedWindow("Original Image");
//    	cv::imshow("Original Image",mat);
//    	cv::waitKey(0);
//    	std::cout << "ok!\n";

    	// ----------------------------------------------------------
    	// opencv BGR format change to RGB
    	// ----------------------------------------------------------
    	cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

        int h = image_size, w = image_size;
        int im_h = mat.rows, im_w = mat.cols, chs = mat.channels();
        float res_aspect_ratio = w*1.0/h;
        float input_aspect_ratio = im_w*1.0/im_h;

        int dif = im_w;
        if( im_h > im_w ) int dif = im_h;

        int interpolation = cv::INTER_CUBIC;
        if( dif > static_cast<int>((h+w)*1.0/2) ) interpolation = cv::INTER_AREA;

        cv::Mat Y;
        if( input_aspect_ratio != res_aspect_ratio ) {
            if( input_aspect_ratio > res_aspect_ratio ) {
                int im_w_r = static_cast<int>(input_aspect_ratio*h);
                int im_h_r = h;
                cv::resize(mat, mat, cv::Size(im_w_r, im_h_r), interpolation);
                int x1 = static_cast<int>((im_w_r - w)/2);
                int x2 = x1 + w;
                mat(cv::Rect(x1, 0, w, im_h_r)).copyTo(Y);
            }

            if( input_aspect_ratio < res_aspect_ratio ) {
                int im_w_r = w;
                int im_h_r = static_cast<int>(w/input_aspect_ratio);
                cv::resize(mat, mat, cv::Size(im_w_r , im_h_r), interpolation);
                int y1 = static_cast<int>((im_h_r - h)/2);
                int y2 = y1 + h;
                mat(cv::Rect(0, y1, im_w_r, h)).copyTo(Y); // startX,startY,cols,rows
            }
        } else {
        	 cv::resize(mat, Y, cv::Size(w, h), interpolation);
        }

        int label = data[index].second;

        torch::Tensor img_tensor = torch::from_blob(Y.data, { Y.channels(), Y.rows, Y.cols }, torch::kByte); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({1}, label);

    	return {img_tensor.clone().to(torch::kFloat32).div_(255.0), label_tensor.clone().to(torch::kInt64)};
    } else {

    	torch::data::Example<> EE;
    	return(EE);
    }
  }

  torch::optional<size_t> size() const {
    return data.size();
  }

 private:
  Data data;
  std::string datasetPath;
  int image_size;
};

std::pair<Data, Data> readInfo( std::string infoFilePath ) {
  Data train, test;

  std::ifstream stream( infoFilePath.c_str());
  assert(stream.is_open());

  long label;
  std::string path, type;

  while (true) {
    stream >> path >> label >> type;
//    std::cout << path << " " << label << " " << type << std::endl;
    if (type == "train")
      train.push_back(std::make_pair(path, label));
    else if (type == "test")
      test.push_back(std::make_pair(path, label));
    else
      assert(false);

    if (stream.eof())
      break;
  }

  std::random_shuffle(train.begin(), train.end());
  std::random_shuffle(test.begin(), test.end());
  return std::make_pair(train, test);
}


torch::Tensor load_image(std::string path) {
    cv::Mat mat;

    //mat = cv::imread("./data/dog.jpg", cv::IMREAD_COLOR);
    mat = cv::imread(path.c_str(), cv::IMREAD_COLOR);

//    cv::imshow("origin BGR image", mat);
//    cv::waitKey(0);
//    cv::destroyAllWindows();

    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    int h = 224, w = 224;

    int im_h = mat.rows, im_w = mat.cols, chs = mat.channels();
    float res_aspect_ratio = w*1.0/h;
    float input_aspect_ratio = im_w*1.0/im_h;

    int dif = im_w;
    if( im_h > im_w ) int dif = im_h;

    int interpolation = cv::INTER_CUBIC;
    if( dif > static_cast<int>((h+w)*1.0/2) ) interpolation = cv::INTER_AREA;

    cv::Mat Y;

    if( input_aspect_ratio != res_aspect_ratio ) {
        if( input_aspect_ratio > res_aspect_ratio ) {
            int im_w_r = static_cast<int>(input_aspect_ratio*h);
            int im_h_r = h;

            cv::resize(mat, mat, cv::Size(im_w_r, im_h_r), (0,0), (0,0), interpolation);
            int x1 = static_cast<int>((im_w_r - w)/2);
            int x2 = x1 + w;
            mat(cv::Rect(x1, 0, w, im_h_r)).copyTo(Y);
        }

        if( input_aspect_ratio < res_aspect_ratio ) {
            int im_w_r = w;
            int im_h_r = static_cast<int>(w/input_aspect_ratio);
            cv::resize(mat, mat, cv::Size(im_w_r , im_h_r), (0,0), (0,0), interpolation);
            int y1 = static_cast<int>((im_h_r - h)/2);
            int y2 = y1 + h;
            mat(cv::Rect(0, y1, im_w_r, h)).copyTo(Y); // startX,startY,cols,rows
        }
    } else {
    	 cv::resize(mat, Y, cv::Size(w, h), interpolation);
    }

    torch::Tensor img_tensor = torch::from_blob(Y.data, {  Y.channels(), Y.rows, Y.cols }, torch::kByte); // Channels x Height x Width
//    img_tensor = img_tensor.permute({ 2, 0, 1 });
    /*
    std::vector<cv::Mat> channels(3);
    cv::split(Y, channels);
    auto R = torch::from_blob(
    	        	        channels[2].data,
    	        	        {Y.rows, Y.cols},
    	        	        torch::kUInt8);
    auto G = torch::from_blob(
    	        	        channels[1].data,
    	        	        {Y.rows, Y.cols},
    	        	        torch::kUInt8);

    auto B = torch::from_blob(
    	        	        channels[0].data,
    	        	        {Y.rows, Y.cols},
    	        	        torch::kUInt8);

    auto img_tensor = torch::cat({R, G, B})
    	        	                     .view({3, Y.rows, Y.cols})
    	        	                     .to(torch::kByte);
*/
    std::cout << "img_tensor1=" << img_tensor.sizes() << std::endl;

//    auto img_tensor = t_tensor.permute({2 , 0, 1}).clone();
//    std::cout << "img_tensor2=" << img_tensor.sizes() << std::endl;

    auto t = img_tensor.to(torch::kFloat).div_(255.0);
    std::cout << "t=" << t.sizes() << std::endl;
/*
    auto tt = t.detach().clone().mul(255).to(torch::kByte);
    auto t4mat = tt.clone().permute({1, 2, 0});

    int width = t4mat.size(0);
    int height = t4mat.size(1);
    cv::Mat mgm(cv::Size{ width, height }, CV_8UC3, t4mat.data_ptr<uchar>());

    cv::cvtColor(mgm, mgm, cv::COLOR_RGB2BGR);
    cv::imshow("converted color image", mgm.clone());
 	cv::waitKey(0);
 	cv::destroyAllWindows();
*/
    return t.clone();
}

void displayImage(std::string f1, std::string f2) {
	torch::manual_seed(0);
	plt::figure_size(800, 500);
	plt::subplot(1, 2, 1);

	torch::Tensor a = load_image(f1);
	torch::Tensor b = load_image(f2);

	torch::Tensor c = torch::stack({a, b}, 0);
/*
	 torch::Tensor a = torch::rand({3,4,4}).mul(255).clamp_max_(255).clamp_min_(0).to(torch::kByte);
//	 a = a.permute({2,0,1});
	 a = a.to(torch::kFloat).div_(255.0);
	 torch::Tensor b = torch::rand({3,4,4}).mul(255).clamp_max_(255).clamp_min_(0).to(torch::kByte);
//	 b = b.permute({2,0,1});
	 b = b.to(torch::kFloat).div_(255.0);
	 torch::Tensor c = torch::stack({a,b},0);

	 std::cout<<a<<std::endl;
	 std::cout<<b<<std::endl;
	 std::cout<<c[0]<<std::endl;
*/
	 a = a.permute({1,2,0}).mul(255).to(torch::kByte);
//	 a = a.to(torch::kByte);
	 std::cout << a.sizes() << std::endl;

	 std::vector<uchar> z(a.size(0) * a.size(1) * a.size(2));
	 std::memcpy(&(z[0]), a.data_ptr<uchar>(),sizeof(uchar)*a.numel());

	 const uchar* zptr = &(z[0]);
	 plt::title("image a");
	 plt::imshow(zptr, a.size(0), a.size(1), a.size(2));

//	 auto aa = c[0].to(torch::kByte); //
	 auto aa = c[0].clone().permute({1,2,0}).mul(255).to(torch::kByte);
	 std::cout << aa.sizes() << std::endl;

	 std::vector<uchar> za(aa.size(0) * aa.size(1) * aa.size(2));
	 std::memcpy(&(za[0]), aa.data_ptr<uchar>(),sizeof(uchar)*aa.numel());

	 const uchar* zptra = &(za[0]);
	 plt::subplot(1, 2, 2);
	 plt::title("image aa");
	 plt::imshow(zptra, aa.size(0), aa.size(1), aa.size(2));
	 plt::show();


	 auto t4mat = c[0].clone().permute({1,2,0}).mul(255).to(torch::kByte);

	 int width = t4mat.size(0);
	 int height = t4mat.size(1);
	 cv::Mat mgm(cv::Size{ width, height }, CV_8UC3, t4mat.data_ptr<uchar>());

	 cv::cvtColor(mgm, mgm, cv::COLOR_RGB2BGR);
	 cv::imshow("converted color image", mgm.clone());
	 cv::waitKey(0);
	 cv::destroyAllWindows();
}

std::vector<unsigned char> tensorToMatrix(torch::Tensor data) {
	auto mimg = data.permute({1,2,0}).mul(255).to(torch::kByte);
	std::vector<unsigned char> z(mimg.size(0) * mimg.size(1) * mimg.size(2));
	std::memcpy(&(z[0]), mimg.data_ptr<unsigned char>(),sizeof(unsigned char)*mimg.numel());
	return z;
}


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	cv::Mat org_img_mat = cv::imread("./data/group.jpg", cv::IMREAD_COLOR);

	std::cout << org_img_mat.size << std::endl;

	if( org_img_mat.empty() ) std::cout << "No image file?\n";

	cv::Mat img_mat = org_img_mat.clone();

	std::cout << img_mat.channels() << std::endl;

	cv::cvtColor(img_mat, img_mat, cv::COLOR_BGR2RGB);

	cv::Mat rgb_mat = img_mat.clone();

	torch::Tensor  img_tensor = torch::from_blob(img_mat.data, {img_mat.rows, img_mat.cols, img_mat.channels()}, torch::kByte);  // {0,1,2} = {H,W,C}
	std::cout << img_tensor.sizes() << std::endl;

	img_tensor = img_tensor.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);  							// {0,1,2} = {H,W,C} ===> {0,1,2} = {C,H,W}
//	std::cout << img_tensor << std::endl;

	std::cout << "After permute and div: " << img_tensor.sizes() << std::endl;

	torch::Tensor data_out = img_tensor.contiguous().detach().clone();

	std::cout << data_out.sizes() << std::endl;

	auto rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});

	std::cout << "Rev permute " << rev_tensor << std::endl;

	std::cout << rev_tensor.sizes() << std::endl;

    // shape of tensor
    int64_t height = rev_tensor.size(0);
    int64_t width = rev_tensor.size(1);

    // Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
    // so we must reshape tensor, otherwise we get a 3x3 grid
    auto tensor = rev_tensor.reshape({width * height * rev_tensor.size(2)});
    std::cout << "tensor.sizes(): " << tensor.sizes() << ", rev_tensor.sizes(): " << rev_tensor.sizes() << std::endl;

    // CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
    std::cout << "CV_8UC3: " << CV_8UC3 << std::endl;
    cv::Mat rev_rgb_mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr());

    cv::Mat rev_bgr_mat = rev_rgb_mat.clone();

	cv::cvtColor(rev_bgr_mat, rev_bgr_mat, cv::COLOR_RGB2BGR);

    // Show Image inside a window with the name provided
    cv::imshow("Original image", org_img_mat);
    // Wait for any keystroke
    cv::waitKey(0);

    cv::imshow("RGB image", rgb_mat);
    cv::waitKey(0);

    cv::imshow("rev RGB image", rev_rgb_mat);
    cv::waitKey(0);

    cv::imshow("rev BGR image", rev_bgr_mat);
    cv::waitKey(0);
    cv::destroyAllWindows();


    plt::figure_size(800, 500);

    // --- mat to tensor
	torch::Tensor  rev_rgb_tensor = torch::from_blob(rev_rgb_mat.data,
				{rev_rgb_mat.rows, rev_rgb_mat.cols, rev_rgb_mat.channels()}, torch::kByte);

	std::cout << "rev_tensor.sizes(): " << rev_tensor.sizes() << ", rev_bgr_tensor.sizes(): " << rev_rgb_tensor.sizes() << std::endl;

	// --- tensor to matrix
    std::vector<uchar> rev_rgb_tensor_z(rev_rgb_tensor.size(0) * rev_rgb_tensor.size(1) * rev_rgb_tensor.size(2));
    std::memcpy(&(rev_rgb_tensor_z[0]), rev_rgb_tensor.data_ptr<uchar>(), sizeof(uchar)*rev_rgb_tensor.numel());

    // --- mat to tensor
	torch::Tensor  rev_bgr_tensor = torch::from_blob(rev_bgr_mat.data,
			{rev_bgr_mat.rows, rev_bgr_mat.cols, rev_bgr_mat.channels()}, torch::kByte);

	// --- tensor to matrix
    std::vector<uchar> rev_bgr_tensor_z(rev_bgr_tensor.size(0) * rev_bgr_tensor.size(1) * rev_bgr_tensor.size(2));
    std::memcpy(&(rev_bgr_tensor_z[0]), rev_bgr_tensor.data_ptr<uchar>(), sizeof(uchar)*rev_bgr_tensor.numel());

    const uchar* zptr = &(rev_rgb_tensor_z[0]);
    plt::subplot(1, 2, 1);
    plt::title("image rev_RGB_tensor");
    plt::imshow(zptr, rev_rgb_tensor.size(0), rev_rgb_tensor.size(1), rev_rgb_tensor.size(2));

    const uchar* zptr2 = &(rev_bgr_tensor_z[0]);
    plt::subplot(1, 2, 2);
    plt::title("image rev_BGR_tensor");
    plt::imshow(zptr2, rev_bgr_tensor.size(0), rev_bgr_tensor.size(1), rev_bgr_tensor.size(2));
    plt::show();

/*
    plt::figure_size(800, 500);
    plt::subplot(1, 2, 1);
    auto batch = *train_data_loader->begin();
    std::vector<unsigned char> z = tensorToMatrix(batch.data[1]);
    const uchar* zptr = &(z[0]);
    int label = batch.target[1].item<int64_t>();
    std::string t = cls[label];
    std::string tlt = flowerLabels[t]; //cls[label];
    plt::title(tlt.c_str());
    plt::imshow(zptr, img_size, img_size, 3);

    plt::subplot(1, 2, 2);
    std::vector<unsigned char> z2 = tensorToMatrix(batch.data[7]);
    const uchar* zptr2 = &(z2[0]);
    label = batch.target[7].item<int64_t>();
    t = cls[label];
    tlt = flowerLabels[t]; //cls[label];
    plt::title(tlt.c_str());
    plt::imshow(zptr2, img_size, img_size, 3);
    plt::show();
*/

    return 0;
}
