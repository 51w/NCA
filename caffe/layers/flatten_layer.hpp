#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
public:
	explicit FlattenLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Flatten"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

};

}  // namespace caffe
