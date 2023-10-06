#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>

#include "Darknet.h"

using namespace std; 
using namespace std::chrono; 

using torch::indexing::Slice;
using torch::indexing::None;


std::vector<std::string> Object_Names(const std::string path) {
    std::vector<std::string> class_names = std::vector<std::string>();

    std::string class_name;
    std::ifstream ifs(path, std::ios::in);
    size_t i = 1;
    if( ! ifs.fail() ) {
    	while( getline(ifs, class_name) ) {
    		if( class_name.length() > 2 ) {
    			class_names.push_back(class_name);
    		}
    	}
    } else {
    	std::cerr << "Error : can't open the class name file." << std::endl;
    	std::exit(1);
    }

    ifs.close();
    // End Processing
    return class_names;
}

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width + 5, -text.height), text_bk_color, cv::FILLED);
    cv::putText(im, label, pt, fontface, scale, text_color, thickness, cv::LINE_AA);
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


int main(int argc, const char* argv[]) {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	std::vector<std::string> objNames = Object_Names("./src/19_ObjectDetection/YOLOv3/models/coco.names");

	std::string filename = "";

    if (argc != 2) {
        //std::cerr << "usage: yolo-app <image path>\n";
        //return -1;
        filename = "./src/19_ObjectDetection/YOLOv3/imgs/giraffe.jpg";
    } else {
    	filename = string(argv[1]);
    }

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {        
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 416;

    Darknet net("./src/19_ObjectDetection/YOLOv3/models/yolov3.cfg", &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    net.load_weights("./src/19_ObjectDetection/YOLOv3/models/yolov3.weights");
    std::cout << "weight loaded ..." << endl;

    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;
    
    cv::Mat origin_image, resized_image;

    // origin_image = cv::imread("../139.jpg");
    origin_image = cv::imread(filename.c_str());
    
    cv::cvtColor(origin_image, resized_image,  cv::COLOR_BGR2RGB);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
    img_tensor = img_tensor.permute({0,3,1,2});

    auto start = std::chrono::high_resolution_clock::now();
       
    auto output = net.forward(img_tensor);

    // filter result by NMS 
    int   class_num = 80;
    float confidence = 0.8;
    float nms_conf = 0.5;
    auto result = net.write_results(output, class_num, confidence, nms_conf);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start); 

    // It should be known that it takes longer time at first time
    std::cout << "inference taken : " << duration.count() << " ms" << endl; 

    if (result.dim() == 1) {
        std::cout << "no object found" << endl;
    }  else  {
        int obj_num = result.size(0);

        std::cout << obj_num << " objects found" << endl;

        float w_scale = float(origin_image.cols) / input_image_size;
        float h_scale = float(origin_image.rows) / input_image_size;

        result.select(1,1).mul_(w_scale);
        result.select(1,2).mul_(h_scale);
        result.select(1,3).mul_(w_scale);
        result.select(1,4).mul_(h_scale);

        //auto result_data = result.accessor<float, 2>();

        for (int i = 0; i < result.size(0); i++) {
        	auto x1 = result.index({i, 1}).item<float>();
        	auto y1 = result.index({i, 2}).item<float>();
        	auto x2 = result.index({i, 3}).item<float>();
        	auto y2 = result.index({i, 4}).item<float>();
        	if( x1 <= 0 ) x1 = 5;
        	if( y1 <= 20 ) y1 = 20;
        	cv::Scalar color = cv::Scalar(0, 0, 255);

            cv::rectangle(origin_image, cv::Point(x1, y1), cv::Point(x2, y2), color, 1, 1, 0);

            int labIdx = (result.index({i, 7})).item<int>();

            if( labIdx > 0 &&  labIdx < objNames.size() ) {

                auto text_color = cv::Scalar(255,255,255);
                if( color == cv::Scalar(255,255,255) ) text_color = cv::Scalar(0,0,0);
                const std::string label = objNames[labIdx] + "=" +
                						  to_string_with_precision((result.index({i, 6})).item<float>(), 4);

                const cv::Point pt = cv::Point(x1, y1-5);
                setLabel(origin_image, label, text_color, color, pt);
            }
        }

        //cv::imwrite("./src/19_ObjectDetection/YOLOv3/out-det.jpg", origin_image);
        cv::imshow("Predicted", origin_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "Done" << endl;
    
    return 0;
}
