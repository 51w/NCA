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

int main(int argc, char** argv) 
{
	string mean_file = "models/imagenet_mean.binaryproto";
	int inputH = 227;
	int inputW = 227;
	Net<float> net(argv[1]);
	net.CopyTrainedLayersFrom(argv[2]);

	Mat mean_;
	cv::Scalar channel_mean = {104.007, 116.669, 122.679, 0};
	mean_ = cv::Mat(inputH, inputW, CV_32FC3, channel_mean);


	Mat img = imread(argv[3]);

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
	cv::resize(img, sample_resized, input_geometry_);

	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC3);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);
	cv::split(sample_normalized, input_channels);


	LOG(INFO) << "start.";
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
		std::cout << ">>" << maxN[i] << "  <-->  " << output[maxN[i]] << "  <-->  " << labels_[maxN[i]] << std::endl; 
	}
}