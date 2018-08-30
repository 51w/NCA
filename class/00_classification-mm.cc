#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "net.hpp"

using namespace cv;
using namespace std;
using namespace caffe;


typedef std::pair<string, float> Prediction;

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

int main(int argc, char** argv) {
	//string mean_file = "ncaffe/models/imagenet_mean.binaryproto";
	//string label_file = argv[4];
	int inputH = 224;
	int inputW = 224;

	string file = "data/panda.jpg";

	Net<float> net(argv[1]);
	net.CopyTrainedLayersFrom(argv[2]);

	//LOG(INFO) << "1111111111111111111";

	Mat mean_;
	cv::Scalar channel_mean = {103.94, 116.78, 123.68, 0};
	//cv::Scalar channel_mean = {103.94f, 116.78f, 123.68f};
	mean_ = cv::Mat(inputH, inputW, CV_32FC3, channel_mean);
	//LOG(INFO) << "222222222222222222";

	Mat img = imread(argv[3]);
	int H = img.rows;
	int W = img.cols;
	int C = img.channels();
	Mat src;
	if(H <= W)
	{
		src = Mat(H, H, CV_8UC3);
		
		int offset = (W - H) / 2;
		for(int i=0; i<H; i++)
		{
			for(int j=0; j<H; j++)
			{
				src.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i,j+offset)[0];
				src.at<Vec3b>(i,j)[1] = img.at<Vec3b>(i,j+offset)[1];
				src.at<Vec3b>(i,j)[2] = img.at<Vec3b>(i,j+offset)[2];
			}
		}
	}
	else
	{
		src = Mat(W, W, CV_8UC3);
		
		int offset = (H - W) / 2;
		for(int i=0; i<W; i++)
		{
			for(int j=0; j<W; j++)
			{
				src.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i+offset,j)[0];
				src.at<Vec3b>(i,j)[1] = img.at<Vec3b>(i+offset,j)[1];
				src.at<Vec3b>(i,j)[2] = img.at<Vec3b>(i+offset,j)[2];
			}
		}
	}
	

	int height = img.rows;
	int width = img.cols;
	Size input_geometry_ = cv::Size(inputH, inputW);
	std::vector<cv::Mat> input_channels;
	shared_ptr<Blob<float> > input_layer = net.blob_by_name("data");
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(inputH, inputW, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += inputH * inputW;
	}
	cv::Mat sample_resized;
	cv::resize(src, sample_resized, input_geometry_);

	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC3);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);
	
	cv::Mat zz(inputH, inputW, CV_32FC3);
	for(int i=0; i<inputH; i++)
	{
		for(int j=0; j<inputW; j++)
		{
			zz.at<Vec3f>(i,j)[0] = sample_normalized.at<Vec3f>(i,j)[0] * 0.017;
			zz.at<Vec3f>(i,j)[1] = sample_normalized.at<Vec3f>(i,j)[1] * 0.017;
			zz.at<Vec3f>(i,j)[2] = sample_normalized.at<Vec3f>(i,j)[2] * 0.017;
		}
	}
	cv::split(zz, input_channels);


	LOG(INFO) << "start net...";
	net.Forward();
	LOG(INFO) << "end.";

	
	vector<string> labels_;
	std::ifstream labels("models/synset_words.txt");
	//labels.open("synset_words.txt");
	CHECK(labels) << "Unable to open labels file synset_words.txt";
	string line;
	while(std::getline(labels, line))
	labels_.push_back(string(line));

	vector<float> output;
	shared_ptr<Blob<float> > result = net.blob_by_name("prob");
	const float* result_data = result->cpu_data();
	for (int i = 0; i < result->count(); ++i)
	{
		//LOG(INFO) << result_data[i];
		output.push_back(result_data[i]);
	}
	LOG(INFO) << "Total: " << output.size();

	vector<int> maxN = Argmax(output, 5);

	for (int i = 0; i < maxN.size(); ++i)
	{
		LOG(INFO) << ">>" << maxN[i] << "  <-->  " << output[maxN[i]] << "  " << labels_[maxN[i]]; 
	}
}