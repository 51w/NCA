#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class PriorBoxLayer : public Layer<Dtype> {
public:
	explicit PriorBoxLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "PriorBox"; }
	virtual inline int ExactBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	vector<float> min_sizes_;
	vector<float> max_sizes_;
	vector<float> aspect_ratios_;
	bool flip_;
	int num_priors_;
	bool clip_;
	vector<float> variance_;

	int img_w_;
	int img_h_;
	float step_w_;
	float step_h_;

	float offset_;
};

}  // namespace caffe