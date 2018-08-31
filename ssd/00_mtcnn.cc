#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "net.hpp"

using namespace cv;
using namespace std;
using namespace caffe;

shared_ptr<Net<float> > PNet_;
shared_ptr<Net<float> > RNet_;
shared_ptr<Net<float> > ONet_;

typedef struct FaceRect {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
} FaceRect;

typedef struct FacePts {
	float x[5], y[5];
} FacePts;

typedef struct FaceInfo {
	FaceRect bbox;
	cv::Vec4f regression;
	FacePts facePts;
	double roll;
	double pitch;
	double yaw;
} FaceInfo;

std::vector<FaceInfo> condidate_rects_;
std::vector<FaceInfo> total_boxes_;
std::vector<FaceInfo> regressed_rects_;
std::vector<FaceInfo> regressed_pading_;

void GenerateBoundingBox(shared_ptr<Blob<float> > confidence, shared_ptr<Blob<float> > reg,
	float scale, float thresh, int image_width, int image_height) {
	int stride = 2;
	int cellSize = 12;

	int curr_feature_map_w_ = std::ceil((image_width - cellSize)*1.0 / stride) + 1;
	int curr_feature_map_h_ = std::ceil((image_height - cellSize)*1.0 / stride) + 1;

	//std::cout << "Feature_map_size:"<< curr_feature_map_w_ <<" "<<curr_feature_map_h_<<std::endl;
	int regOffset = curr_feature_map_w_*curr_feature_map_h_;
	// the first count numbers are confidence of face
	int count = confidence->count() / 2;
	const float* confidence_data = confidence->cpu_data();
	confidence_data += count;
	const float* reg_data = reg->cpu_data();

	condidate_rects_.clear();
	for (int i = 0; i<count; i++) {
		if (*(confidence_data + i) >= thresh) {
			int y = i / curr_feature_map_w_;
			int x = i - curr_feature_map_w_ * y;

			float xTop = (int)((x*stride + 1) / scale);
			float yTop = (int)((y*stride + 1) / scale);
			float xBot = (int)((x*stride + cellSize - 1 + 1) / scale);
			float yBot = (int)((y*stride + cellSize - 1 + 1) / scale);
			FaceRect faceRect;
			faceRect.x1 = xTop;
			faceRect.y1 = yTop;
			faceRect.x2 = xBot;
			faceRect.y2 = yBot;
			faceRect.score = *(confidence_data + i);
			FaceInfo faceInfo;
			faceInfo.bbox = faceRect;
			faceInfo.regression = cv::Vec4f(reg_data[i + 0 * regOffset], reg_data[i + 1 * regOffset], reg_data[i + 2 * regOffset], reg_data[i + 3 * regOffset]);
			condidate_rects_.push_back(faceInfo);
		}
	}
}

bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
	return a.bbox.score > b.bbox.score;
}

std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes,
	float thresh, char methodType) {
	std::vector<FaceInfo> bboxes_nms;
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}

		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		FaceRect select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
		float x1 = static_cast<float>(select_bbox.x1);
		float y1 = static_cast<float>(select_bbox.y1);
		float x2 = static_cast<float>(select_bbox.x2);
		float y2 = static_cast<float>(select_bbox.y2);

		select_idx++;
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			FaceRect& bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}

void Padding(int img_w, int img_h) {
	for (int i = 0; i<regressed_rects_.size(); i++) {
		FaceInfo tempFaceInfo;
		tempFaceInfo = regressed_rects_[i];
		tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_w) ? img_w : regressed_rects_[i].bbox.y2;
		tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_h) ? img_h : regressed_rects_[i].bbox.x2;
		tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 <1) ? 1 : regressed_rects_[i].bbox.y1;
		tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 <1) ? 1 : regressed_rects_[i].bbox.x1;
		regressed_pading_.push_back(tempFaceInfo);
	}
}

void Bbox2Square(std::vector<FaceInfo>& bboxes) {
	for (int i = 0; i<bboxes.size(); i++) {
		float h = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
		float w = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
		float side = h>w ? h : w;
		bboxes[i].bbox.x1 += (h - side)*0.5;
		bboxes[i].bbox.y1 += (w - side)*0.5;

		bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
		bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
		bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
		bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

	}
}

std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo>& faceInfo, int stage) {
	std::vector<FaceInfo> bboxes;
	for (int bboxId = 0; bboxId<faceInfo.size(); bboxId++) {
		FaceRect faceRect;
		FaceInfo tempFaceInfo;
		float regw = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1;
		regw += (stage == 1) ? 0 : 1;
		float regh = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1;
		regh += (stage == 1) ? 0 : 1;
		faceRect.y1 = faceInfo[bboxId].bbox.y1 + regw * faceInfo[bboxId].regression[0];
		faceRect.x1 = faceInfo[bboxId].bbox.x1 + regh * faceInfo[bboxId].regression[1];
		faceRect.y2 = faceInfo[bboxId].bbox.y2 + regw * faceInfo[bboxId].regression[2];
		faceRect.x2 = faceInfo[bboxId].bbox.x2 + regh * faceInfo[bboxId].regression[3];
		faceRect.score = faceInfo[bboxId].bbox.score;

		tempFaceInfo.bbox = faceRect;
		tempFaceInfo.regression = faceInfo[bboxId].regression;
		if (stage == 3)
			tempFaceInfo.facePts = faceInfo[bboxId].facePts;
		bboxes.push_back(tempFaceInfo);
	}
	return bboxes;
}

void ClassifyFace(const vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,
	shared_ptr<Net<float> >& net, double thresh, char netName) 
{
	int numBox = regressed_rects.size();
	//Blob<float>* crop_input_layer = net->input_blobs()[0];
	shared_ptr<Blob<float> > crop_input_layer = net->blob_by_name("data");
	int input_channels = crop_input_layer->channels();
	int input_width = crop_input_layer->width();
	int input_height = crop_input_layer->height();
	crop_input_layer->Reshape(1, input_channels, input_width, input_height);
	net->Reshape();


	condidate_rects_.clear();
	for (int i = 0; i<numBox; i++) 
	{
		std::vector<cv::Mat> channels;
		float* input_data = crop_input_layer->mutable_cpu_data();
		for (int i = 0; i < crop_input_layer->channels(); ++i) {
			cv::Mat channel(input_height, input_width, CV_32FC1, input_data);
			channels.push_back(channel);
			input_data += input_width * input_height;
		}

		int pad_top = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
		int pad_left = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
		int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
		int pad_bottom = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

		cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1 - 1, regressed_pading_[i].bbox.y2),
			cv::Range(regressed_pading_[i].bbox.x1 - 1, regressed_pading_[i].bbox.x2));
		cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));

		cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_NEAREST);
		crop_img = (crop_img - 127.5)*0.0078125;
		cv::split(crop_img, channels);

		net->Forward();

		int reg_id = 0;
		int confidence_id = 1;
		if (netName == 'o') confidence_id = 2;
		//const Blob<float>* reg = net->output_blobs()[reg_id];
		//const Blob<float>* confidence = net->output_blobs()[confidence_id];
		//const Blob<float>* points_offset = net->output_blobs()[1];

		string aa;
		aa = "conv5-2";
		if (netName == 'o')
		{
			aa = "conv6-2";
		}


		shared_ptr<Blob<float> > reg = net->blob_by_name(aa);
		shared_ptr<Blob<float> > confidence = net->blob_by_name("prob1");

		
		//shared_ptr<Blob<float> > points_offset = net->blob_by_name("conv6-3");
		

		const float* confidence_data = confidence->cpu_data() + confidence->count() / 2;
		const float* reg_data = reg->cpu_data();
		const float* points_data;
		if (netName == 'o') {
			shared_ptr<Blob<float> > points_offset = net->blob_by_name("conv6-3");
			points_data = points_offset->cpu_data();
		}

		if (*(confidence_data) > thresh) {
			FaceRect faceRect;
			faceRect.x1 = regressed_rects[i].bbox.x1;
			faceRect.y1 = regressed_rects[i].bbox.y1;
			faceRect.x2 = regressed_rects[i].bbox.x2;
			faceRect.y2 = regressed_rects[i].bbox.y2;
			faceRect.score = *(confidence_data);
			FaceInfo faceInfo;
			faceInfo.bbox = faceRect;
			faceInfo.regression = cv::Vec4f(reg_data[0], reg_data[1], reg_data[2], reg_data[3]);

			// x x x x x y y y y y
			if (netName == 'o') {
				FacePts face_pts;
				float w = faceRect.y2 - faceRect.y1 + 1;
				float h = faceRect.x2 - faceRect.x1 + 1;
				for (int j = 0; j<5; j++) {
					face_pts.y[j] = faceRect.y1 + *(points_data + j) * h - 1;
					face_pts.x[j] = faceRect.x1 + *(points_data + j + 5) * w - 1;
				}
				faceInfo.facePts = face_pts;
			}
			condidate_rects_.push_back(faceInfo);
		}
	}
	regressed_pading_.clear();
}

void Detect(const Mat& image, vector<FaceInfo>& faceInfo, int minSize, double* threshold, double factor)
{
	Mat sample_single, resized;
	image.convertTo(sample_single, CV_32FC3);
	cvtColor(sample_single, sample_single, cv::COLOR_BGR2RGB);
	sample_single = sample_single.t();

	int height = image.rows;
	int width = image.cols;
	int minWH = std::min(height, width);
	int factor_count = 0;
	double m = 12. / minSize;
	minWH *= m;
	vector<double> scales;
	while (minWH >= 12)
	{
		scales.push_back(m * std::pow(factor, factor_count));
		minWH *= factor;
		++factor_count;
	}

	shared_ptr<Blob<float> > input_layer = PNet_->blob_by_name("data");
	for (int i = 0; i<factor_count; i++)
	{
		double scale = scales[i];
		int ws = std::ceil(height*scale);
		int hs = std::ceil(width*scale);

		//cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_AREA);
		cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_NEAREST);
		resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5*0.0078125);

		// input data
		input_layer->Reshape(1, 3, hs, ws);
		PNet_->Reshape();

		std::vector<cv::Mat> input_channels;
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(hs, ws, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += hs * ws;
		}
		cv::split(resized, input_channels);

		PNet_->Forward();

		shared_ptr<Blob<float> > reg = PNet_->blob_by_name("conv4-2");
		shared_ptr<Blob<float> > confidence = PNet_->blob_by_name("prob1");

		std::cout << ">>" << confidence->count() << std::endl;

		GenerateBoundingBox(confidence, reg, scale, threshold[0], ws, hs);
		std::vector<FaceInfo> bboxes_nms = NonMaximumSuppression(condidate_rects_, 0.6, 'u');
		total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
	}
	
	printf("total_boxes_.size()=%ld\n", total_boxes_.size());
	std::cout << "2> " << total_boxes_[0].bbox.x1 << " " << total_boxes_[0].bbox.x2 << " "
		<< total_boxes_[0].bbox.y1 << " " << total_boxes_[0].bbox.y2 << " "
		<< std::endl;

	int numBox = total_boxes_.size();
	if (numBox != 0) {
		total_boxes_ = NonMaximumSuppression(total_boxes_, 0.7, 'u');
		regressed_rects_ = BoxRegress(total_boxes_, 1);
		total_boxes_.clear();

		Bbox2Square(regressed_rects_);
		Padding(width, height);


		/// Second stage
		ClassifyFace(regressed_rects_, sample_single, RNet_, threshold[1], 'r');

		condidate_rects_ = NonMaximumSuppression(condidate_rects_, 0.7, 'u');
		regressed_rects_ = BoxRegress(condidate_rects_, 2);

		Bbox2Square(regressed_rects_);
		Padding(width, height);


		/// three stage
		numBox = regressed_rects_.size();
		std::cout << "3> " << numBox << std::endl;
		if (numBox != 0) {

			ClassifyFace(regressed_rects_, sample_single, ONet_, threshold[2], 'o');

			regressed_rects_ = BoxRegress(condidate_rects_, 3);

			std::cout << "4> " << regressed_rects_.size() << std::endl;
			faceInfo = NonMaximumSuppression(regressed_rects_, 0.7, 'm');
		}
	}
	regressed_pading_.clear();
	regressed_rects_.clear();
	condidate_rects_.clear();
}

int main(int argc, char *argv[])
{
	union
	{
		struct
		{
			unsigned char f0;
			unsigned char f1;
			unsigned char f2;
			unsigned char f3;
		};
		unsigned int tag;
	} flag_struct;

	LOG(INFO) << sizeof(flag_struct);

	PNet_.reset(new Net<float>("models/mtcnn/det1.prototxt") );
	PNet_->CopyTrainedLayersFrom("models/mtcnn/det1.caffemodel");

	RNet_.reset(new Net<float>("models/mtcnn/det2.prototxt"));
	RNet_->CopyTrainedLayersFrom("models/mtcnn/det2.caffemodel");

	ONet_.reset(new Net<float>("models/mtcnn/det3.prototxt"));
	ONet_->CopyTrainedLayersFrom("models/mtcnn/det3.caffemodel");

	// param
	double threshold[3] = { 0.7,0.8,0.7 };
	double factor = 0.709;
	int minSize = 80;
	//Mat image = cv::imread(argv[1]);
VideoCapture cam(argv[1]);
Mat image;

while (1) {
cam >> image;
if (image.empty()) break;

	vector<FaceInfo> faceInfo;

	Detect(image, faceInfo, minSize, threshold, factor);

	std::cout << "5> " << faceInfo.size() << std::endl;


	for (int i = 0; i<faceInfo.size(); i++) {
		float x = faceInfo[i].bbox.x1;
		float y = faceInfo[i].bbox.y1;
		float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
		float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
		cv::rectangle(image, cv::Rect(y, x, w, h), cv::Scalar(255, 0, 0), 2);
	}
	for (int i = 0; i<faceInfo.size(); i++) {
		FacePts facePts = faceInfo[i].facePts;
		for (int j = 0; j<5; j++)
			cv::circle(image, cv::Point(facePts.y[j], facePts.x[j]), 1, cv::Scalar(255, 255, 0), 2);
	}
	cv::imshow("MTCNN", image);
	//cv::imwrite("result.jpg", image);
	cv::waitKey(30);
}
	return 0;
}