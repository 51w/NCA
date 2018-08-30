#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

#include "layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class PReLULayer : public NeuronLayer<Dtype> {
public:
	explicit PReLULayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "PReLU"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	bool channel_shared_;
	Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
	Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
	Blob<Dtype> bottom_memory_;  // memory for in-place computation
};

}  // namespace caffe