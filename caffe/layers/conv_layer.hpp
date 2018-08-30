#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

#include "layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
public:
	explicit ConvolutionLayer(const LayerParameter& param)
		: BaseConvolutionLayer<Dtype>(param) {}

	virtual inline const char* type() const { return "Convolution"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline bool reverse_dimensions() { return false; }
	virtual void compute_output_shape();
};

}  // namespace caffe