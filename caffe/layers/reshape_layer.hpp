#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ReshapeLayer : public Layer<Dtype> {
public:
	explicit ReshapeLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Reshape"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {}


	/// vector of axes indices whose dimensions we'll copy from the bottom
	vector<int> copy_axes_;
	/// the index of the axis whose dimension we infer, or -1 if none
	int inferred_axis_;
	/// the product of the "constant" output dimensions
	int constant_count_;
};

}  // namespace caffe