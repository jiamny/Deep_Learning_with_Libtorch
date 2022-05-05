#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;
string labelFile = "./data/models/labelmap_det.txt";
string modelFile = "./data/models/MobileNetSSD_deploy.caffemodel";
string model_text_file = "./data/models/MobileNetSSD_deploy.prototxt";

string objNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv) {
	Mat frame = imread("./data/person.jpg");
	if (frame.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", cv::WINDOW_AUTOSIZE);
	imshow("input image", frame);

	Net net = readNetFromCaffe(model_text_file, modelFile);
	Mat blobImage = blobFromImage(frame, 0.007843,
		Size(300, 300),
		Scalar(127.5, 127.5, 127.5), true, false);
	printf("blobImage width : %d, height: %d\n", blobImage.cols, blobImage.rows);

	net.setInput(blobImage, "data");
	Mat detection = net.forward("detection_out");
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	float confidence_threshold = 0.4;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
			float br_x = detectionMat.at<float>(i, 5) * frame.cols;
			float br_y = detectionMat.at<float>(i, 6) * frame.rows;

			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(frame, object_box, Scalar(0, 0, 255), 1, 8, 0);
			putText(frame, format("%s", objNames[objIndex].c_str()), Point(tl_x, tl_y - 5), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 0), 1);
		}
	}
	imshow("ssd-demo", frame);
	waitKey(0);
	destroyAllWindows();

	return 0;
}



