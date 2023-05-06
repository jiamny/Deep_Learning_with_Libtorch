
#ifndef SRC_23_VEHICLE_AND_TRAFFIC_LANE_DETECTION_YOLO5_H_
#define SRC_23_VEHICLE_AND_TRAFFIC_LANE_DETECTION_YOLO5_H_

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <torch/script.h> // One-stop header.

#include <fstream>
#include <opencv2/opencv.hpp>


const std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
		 cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;


struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

std::vector<std::string> load_class_list(std::string classfilePath);

void load_net(cv::dnn::Net &net, bool is_cuda, std::string onnxFile);

cv::Mat format_yolov5(const cv::Mat &source);

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt);

void detect_objects(cv::Mat& frame, cv::dnn::Net net, std::vector<std::string> class_list);

#endif /* SRC_23_VEHICLE_AND_TRAFFIC_LANE_DETECTION_YOLO5_H_ */
