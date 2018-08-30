#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

#include "layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
public:
	explicit DropoutLayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Dropout"; }

protected:

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	Blob<unsigned int> rand_vec_;
	/// the probability @f$ p @f$ of dropping any input
	Dtype threshold_;
	/// the scale for undropped inputs at train time @f$ 1 / (1 - p) @f$
	Dtype scale_;
	unsigned int uint_thres_;
};

}  // namespace caffe