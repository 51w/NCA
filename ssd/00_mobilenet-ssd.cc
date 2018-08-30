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
	//Net<float> net("models/MobileNetSSD_deploy.prototxt");
	//net.CopyTrainedLayersFrom("models/MobileNetSSD_deploy.caffemodel");
	Net<float> net(argv[1]);
	net.CopyTrainedLayersFrom(argv[2]);
	//Net<float> net("models/squeezenet_ssd_voc_deploy.prototxt");
	//net.CopyTrainedLayersFrom("models/squeezenet_iter_74000.caffemodel");
	Mat img = imread(argv[3]);
	
// VideoCapture cam(argv[3]);
// Mat img;

// while (1) {
// cam >> img;
// if (img.empty()) break;


	const float kScoreThreshold = 0.01f;
	const char* kClassNames[] = { "__background__", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };

	int height = img.rows;
	int width = img.cols;
	Mat imgResized;
	cv::resize(img, imgResized, Size(300, 300));

	vector<Mat> bgr;
	cv::split(imgResized, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1.f / 127.5f, -1.f);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f / 127.5f, -1.f);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f / 127.5f, -1.f);

	shared_ptr<Blob<float> > data = net.blob_by_name("data");
	data->Reshape(1, 3, imgResized.rows, imgResized.cols);
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias * sizeof(float);
	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);

	LOG(INFO) << "start net...";
	net.Forward();
	LOG(INFO) << "end.";

	// get output, shape is N x 7
	shared_ptr<Blob<float> > result = net.blob_by_name("detection_out");
	const float* result_data = result->cpu_data();


	// for (int i=0; i < net.blobs_.size(); i++)
	// {
		// shared_ptr<Blob<float> > aa = net.blobs_[i];
		// LOG(INFO) << aa->channels() << " " << aa->height() << " " << aa->width();
		// LOG(INFO) << "******************" << aa->count();
	// }
	// for (int i = 0; i < result->count(); ++i)
	// {
		// LOG(INFO) << result_data[i];
	// }
	//LOG(INFO) << result->count() << "   " << result->height();

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
			
			LOG(INFO) << bbox.x1 << " " << bbox.y1 << " " << bbox.x2-bbox.x1 << " " << bbox.y2-bbox.y1;
		}
		result_data += 7;
	}

	// draw
	for (auto& bbox : detections) {
		cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1);
		cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
		char buff[300];
		sprintf(buff, "%s: %.2f", kClassNames[bbox.label], bbox.score);
		cv::putText(img, buff, cv::Point(bbox.x1, bbox.y1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
	}

	//cv::imwrite("ssd-result.jpg", img);
	cv::imshow("result", img);
	cv::waitKey();

//}
	return 0;
}