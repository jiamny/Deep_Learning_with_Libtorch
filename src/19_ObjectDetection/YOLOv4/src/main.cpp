#include"Detector.h"

int main()
{

	cv::Mat image = cv::imread("./src/19_ObjectDetection/YOLOv4/dataset/val/images/2007_005331.jpg");
	Detector detector;
	// gpu id >= 0, otherwise -1 for CPU
	detector.Initialize(-1, 416, 416, "./src/19_ObjectDetection/YOLOv4/dataset/voc_classes.txt");
	// detector.Train("dataset", ".jpg", 30,
	// 	4, 0.001, "weights/detector.pt", "weights/yolo4_tiny.pt");

	detector.LoadWeight("./src/19_ObjectDetection/YOLOv4/weights/detector.pt");

	//detector.Predict(image, true, 0.1);
	detector.Predict(image, true, 0.1, 0.3);

	//speed test
	int64 start = cv::getTickCount();
	int loops = 10;
	for (int i = 0; i < loops; i++) {
		detector.Predict(image, false);
	}
	double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
	std::cout << duration/ loops <<" s per prediction" << std::endl;

	return 0;
}
