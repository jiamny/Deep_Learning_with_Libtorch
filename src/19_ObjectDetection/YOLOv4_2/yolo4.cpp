#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("./src/19_ObjectDetection/YOLOv4_2/dog.jpg");

    if( img.empty() ) {
        std::cout << "dog.jpg not read successfully" << std::endl;
        exit(-1);
    }

    std::vector<std::string> classes;
    std::ifstream file;

    file.open("./src/19_ObjectDetection/YOLOv4_2/coco.names", std::ios_base::in);
    // Exit if file not opened successfully
    if( !file.is_open() ) {
        std::cout << "coco.names not read successfully" << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(file, line)) {
        classes.push_back(line);
    }

    cv::dnn::Net net = cv::dnn::readNetFromDarknet("./src/19_ObjectDetection/YOLOv4_2/yolov4.cfg",
    							 "./src/19_ObjectDetection/YOLOv4_2/yolov4.weights");

    cv::dnn::DetectionModel model = cv::dnn::DetectionModel(net);
    model.setInputParams(1 / 255.0, cv::Size(416, 416), cv::Scalar(), true);

    std::vector<int> classIds;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    model.detect(img, classIds, scores, boxes, 0.6, 0.4);

    for (int i = 0; i < classIds.size(); i++) {
        cv::rectangle(img, boxes[i], cv::Scalar(0, 0, 255), 2);

        char text[100];
        std::snprintf(text, sizeof(text), "%s: %.2f", classes[classIds[i]].c_str(), scores[i]);
        cv::putText(img, text, cv::Point(boxes[i].x, boxes[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("Image", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}



