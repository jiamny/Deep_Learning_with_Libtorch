#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

String haar_data_file = "./data/models/haarcascade_frontalface_alt_tree.xml";
String lbp_data_file = "./data/models/lbpcascade_frontalface_improved.xml";

void faceDemo();

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector
{
public:
	CascadeDetectorAdapter(Ptr<CascadeClassifier> detector) :
		IDetector(),
		Detector(detector)
	{
		CV_Assert(detector);
	}

	void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
	{
		Detector->detectMultiScale(Image, objects, scaleFactor, 1, 0, Size(100, 100), Size(400, 400));
	}

	virtual ~CascadeDetectorAdapter()
	{}

private:
	CascadeDetectorAdapter();
	Ptr<CascadeClassifier> Detector;
};

int main(int argc, char** argv) {
	faceDemo();
	waitKey(0);
	destroyAllWindows();

	return 0;
}

void faceDemo() {
	Mat frame, gray;
	VideoCapture capture;
	capture.open("./data/facerecog.mp4");
	namedWindow("input", cv::WINDOW_AUTOSIZE);
	String cascadeFrontalfilename = "./data/models/haarcascade_frontalface_alt_tree.xml";
	DetectionBasedTracker::Parameters params;

	Ptr<CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);

	cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);

	DetectionBasedTracker tracker(MainDetector, TrackingDetector, params);
	vector<Rect> faces;
	while (capture.read(frame)) {
		cvtColor(frame, gray, COLOR_RGB2GRAY);
		tracker.process(gray);
		tracker.getObjects(faces);
		if (faces.size()) {
			for (size_t i = 0; i < faces.size(); i++) {
				rectangle(frame, faces[i], Scalar(0, 0, 255));
			}
		}
		imshow("input", frame);
		char c = waitKey(10);
		if (c == 27) {
			break;
		}
	}
	tracker.stop();

}
