#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"
#include "util/bbox_util.hpp"


namespace caffe {

template <typename Dtype>
class DetectionOutputLayer : public Layer<Dtype> {
public:
	explicit DetectionOutputLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "DetectionOutput"; }
	virtual inline int MinBottomBlobs() const { return 3; }
	virtual inline int MaxBottomBlobs() const { return 4; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	int num_classes_;
	bool share_location_;
	int num_loc_classes_;
	int background_label_id_;
	CodeType code_type_;
	bool variance_encoded_in_target_;
	int keep_top_k_;
	float confidence_threshold_;

	int num_;
	int num_priors_;

	float nms_threshold_;
	int top_k_;
	float eta_;

	string output_directory_;
	string output_name_prefix_;
	string output_format_;
	map<int, string> label_to_name_;
	map<int, string> label_to_display_name_;
	vector<string> names_;
	vector<pair<int, int> > sizes_;
	int num_test_image_;
	int name_count_;
	bool has_resize_;
	ResizeParameter resize_param_;


	Blob<Dtype> bbox_preds_;
	Blob<Dtype> bbox_permute_;
	Blob<Dtype> conf_permute_;
};

}  // namespace caffe