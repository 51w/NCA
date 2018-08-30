#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void Permute(const int count, Dtype* bottom_data, const bool forward,
	const int* permute_order, const int* old_steps, const int* new_steps,
	const int num_axes, Dtype* top_data);

template <typename Dtype>
class PermuteLayer : public Layer<Dtype> {
public:
	explicit PermuteLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Permute"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	int num_axes_;
	bool need_permute_;

	// Use Blob because it is convenient to be accessible in .cu file.
	Blob<int> permute_order_;
	Blob<int> old_steps_;
	Blob<int> new_steps_;
};

}  // namespace caffe