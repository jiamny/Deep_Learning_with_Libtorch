
#ifndef SRC_23_VEHICLE_AND_TRAFFIC_LANE_DETECTION_YOLO5_H_
#define SRC_23_VEHICLE_AND_TRAFFIC_LANE_DETECTION_YOLO5_H_

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <iomanip>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;


const vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
		 cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255)};

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt);

vector<Mat> pre_process(Mat &input_image, Net &net);

Mat post_process(Mat input_image, vector<Mat> &outputs, const vector<string> &class_name);

#endif /* SRC_23_VEHICLE_AND_TRAFFIC_LANE_DETECTION_YOLO5_H_ */
