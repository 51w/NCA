#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

#include "layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class PowerLayer : public NeuronLayer<Dtype> {
public:

	explicit PowerLayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Power"; }

protected:

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	Dtype power_;
	/// @brief @f$ \alpha @f$ from layer_param_.power_param()
	Dtype scale_;
	/// @brief @f$ \beta @f$ from layer_param_.power_param()
	Dtype shift_;
	/// @brief Result of @f$ \alpha \gamma @f$
	Dtype diff_scale_;
};

} 