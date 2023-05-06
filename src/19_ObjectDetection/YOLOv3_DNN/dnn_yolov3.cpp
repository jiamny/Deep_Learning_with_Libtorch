#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

void image_detection();
void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt);
std::string yolo_cfg = "/media/stree/localssd/DL_data/models/yolov3.cfg";
std::string yolo_model = "/media/stree/localssd/DL_data/models/yolov3.weights";
std::string yolo_tiny_model = "/media/stree/localssd/DL_data/models/yolov3-tiny.weights";
std::string yolo_tiny_cfg = "/media/stree/localssd/DL_data/models/yolov3-tiny.cfg";

bool useDetectionModel = false;

int main(int argc, char** argv) {
	image_detection();
}

void image_detection() {
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(yolo_cfg, yolo_model);

	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
		printf("output layer name : %s\n", outNames[i].c_str());
	}

	std::vector<std::string> classNamesVec;
	std::ifstream classNamesFile("/media/stree/localssd/DL_data/models/coco.names");

	if (classNamesFile.is_open()) {
		std::string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	cv::Mat frame = cv::imread("./data/person.jpg");

	if( useDetectionModel ) {
		cv::dnn::DetectionModel model = cv::dnn::DetectionModel(net);
		model.setInputParams(1 / 255.0, cv::Size(416, 416), cv::Scalar(), true);

		std::vector<int> classIds;
		std::vector<float> scores;
		std::vector<cv::Rect> boxes;
		model.detect(frame, classIds, scores, boxes, 0.6, 0.4);

		for (int i = 0; i < classIds.size(); i++) {
		    cv::rectangle(frame, boxes[i], cv::Scalar(0, 0, 255), 1);

		    char text[100];
		    std::snprintf(text, sizeof(text), "%s: %.2f", classNamesVec[classIds[i]].c_str(), scores[i]);

		    const cv::Point pt = cv::Point(boxes[i].x, boxes[i].y - 5);
		    setLabel(frame, text, cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 255), pt);
		}

	} else {
		cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1 / 255.f, cv::Size(416, 416), cv::Scalar(), true, false);
		net.setInput(inputBlob);
		std::vector<cv::Mat> outs;
		net.forward(outs, outNames);

		std::vector<double> layersTimings;
		double freq = cv::getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		std::ostringstream ss;
		ss << "detection time: " << time << " ms";
		cv::putText(frame, ss.str(), cv::Point(20, 20), 0, 0.5, cv::Scalar(0, 0, 255));

		std::vector<cv::Rect> boxes;
		std::vector<int> classIds;
		std::vector<float> confidences;

		for (size_t i = 0; i<outs.size(); ++i) {
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > 0.5)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}

		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
		for (size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			cv::Rect box = boxes[idx];
			std::string className = classNamesVec[classIds[idx]];
			// box around detected object
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, 8, 0);

			char text[100];
			std::snprintf(text, sizeof(text), "%s: %.2f", className.c_str(), confidences[i]);
			// lable the detected object
			const cv::Point pt = cv::Point(box.tl().x, box.tl().y - 5);
			setLabel(frame, text, cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 255), pt);
		}
	}

	cv::imshow("YOLOv3-Detections", frame);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return;
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
