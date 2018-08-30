#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

#include "layers/bias_layer.hpp"

namespace caffe {

template <typename Dtype>
class ScaleLayer : public Layer<Dtype> {
public:
	explicit ScaleLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Scale"; }
	// Scale
	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MaxBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	shared_ptr<Layer<Dtype> > bias_layer_;
	vector<Blob<Dtype>*> bias_bottom_vec_;
	vector<bool> bias_propagate_down_;
	int bias_param_id_;

	Blob<Dtype> sum_multiplier_;
	Blob<Dtype> sum_result_;
	Blob<Dtype> temp_;
	int axis_;
	int outer_dim_, scale_dim_, inner_dim_;
};


}  // namespace caffe