#pragma once
#include <vector>

#include "layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
public:
	explicit ReLULayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param) {}

	virtual inline const char* type() const { return "ReLU"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

};

}  // namespace caffe