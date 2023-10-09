// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <iomanip>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

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

const std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
		 cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255)};

vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width + 5, - text.height), text_bk_color, cv::FILLED);
    cv::putText(im, label, pt, fontface, scale, text_color, thickness, cv::LINE_AA);
}


Mat post_process(Mat input_image, vector<Mat> &outputs, const vector<string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        const auto color = colors[class_ids[idx] % colors.size()];

        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), color, 1*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ": " + label;

        // Draw class labels.
        setLabel(input_image, label.c_str(), cv::Scalar(0, 0, 0), color,
                                    		   cv::Point(box.x, box.y - 4));
    }
    return input_image;
}


int main(int argc, char **argv) {

	// Load class list.
	vector<string> class_list;
	ifstream ifs("./src/19_ObjectDetection/YOLOv5/config_files/classes.txt");
	string line;

	while (getline(ifs, line)) {
	    class_list.push_back(line);
	}

    cv::Mat frame;
    cv::VideoCapture capture("/media/hhj/localssd/DL_data/videos/sample.mp4");
    //cv::VideoCapture capture("./data/videos/project_video.mp4");
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file\n";
        return -1;
    }


    // Load model.
    Net net;
    net = readNet("/media/hhj/localssd/DL_data/weights/yolo5/yolov5s.onnx");

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while(true)  {

        capture.read(frame);
        if (frame.empty()) {
            std::cout << "End of stream\n";
            break;
        }

        vector<Mat> detections;
        detections = pre_process(frame, net);

        Mat img = post_process(frame.clone(), detections, class_list);

        frame_count++;
        total_frames++;


        if (frame_count >= 30) {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0) {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(img, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", img);

        if (cv::waitKey(1) != -1) {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }

    std::cout << "Total frames: " << total_frames << "\n";

    capture.release();

    return 0;
}
