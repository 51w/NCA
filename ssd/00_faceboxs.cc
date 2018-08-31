#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "net.hpp"

using namespace cv;
using namespace std;
using namespace caffe;

struct BBox {
	float x1, y1, x2, y2, score;
	int label;
};

int main(int argc, char** argv)
{
	
	Net<float> net("models/faceboxes_deploy.prototxt");
	net.CopyTrainedLayersFrom("models/FaceBoxes_1024x1024.caffemodel");

	
	Mat img = imread(argv[1]);
// cv::VideoCapture cam(argv[1]);
// cv::Mat img;

// while (1) {
// cam >>img;
// if (img.empty()) break;	
	
	
	const float kScoreThreshold = 0.1f;
	int height = img.rows;
	int width = img.cols;
	Mat imgResized;
	cv::resize(img, imgResized, Size(1024, 1024));

	int inputH = 1024;
	int inputW = 1024;
	std::vector<cv::Mat> input_channels;
	shared_ptr<Blob<float> > input_layer = net.blob_by_name("data");
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(inputH, inputW, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += inputH * inputW;
	}
	
	Mat mean_;
	Scalar channel_mean = {104.0f, 117.0f, 123.0f, 0};
	mean_ = cv::Mat(inputH, inputW, CV_32FC3, channel_mean);
	
	cv::Mat sample_float;
	imgResized.convertTo(sample_float, CV_32FC3);
	
	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);
	
	cv::split(sample_normalized, input_channels);
	
	LOG(INFO) << "start net...";
	net.Forward();
	LOG(INFO) << "end.";

	// get output, shape is N x 7
	shared_ptr<Blob<float> > result = net.blob_by_name("detection_out");
	const float* result_data = result->cpu_data();


	for (int i = 0; i < result->count(); ++i)
	{
		LOG(INFO) << result_data[i];
	}
	//LOG(INFO) << result->count() << "   " << result->height() << "********************";

	const int num_det = result->height();
	vector<BBox> detections;
	for (int k = 0; k < num_det; ++k) {
		if (result_data[0] != -1 && result_data[2] > kScoreThreshold) {
			// [image_id, label, score, xmin, ymin, xmax, ymax]
			BBox bbox;
			bbox.x1 = result_data[3] * width;
			bbox.y1 = result_data[4] * height;
			bbox.x2 = result_data[5] * width;
			bbox.y2 = result_data[6] * height;
			bbox.score = result_data[2];
			bbox.label = static_cast<int>(result_data[1]);
			detections.push_back(bbox);
			LOG(INFO) << bbox.label << " " << bbox.x1 << " " << bbox.y1 << " " << bbox.x2-bbox.x1 << " " << bbox.y2-bbox.y1;
		}
		result_data += 7;
	}

	// draw
	for (auto& bbox : detections) {
		cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1);
		cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
	}

	//cv::imwrite("faceboxs-result.jpg", img);
	cv::imshow("SSD", img);
	cv::waitKey();
//}
	return 0;
}