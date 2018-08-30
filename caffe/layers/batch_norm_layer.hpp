#pragma once
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BatchNormLayer : public Layer<Dtype> {
public:
	explicit BatchNormLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "BatchNorm"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	Blob<Dtype> mean_, variance_, temp_, x_norm_;
	bool use_global_stats_;
	Dtype moving_average_fraction_;
	int channels_;
	Dtype eps_;

	// extra temporarary variables is used to carry out sums/broadcasting
	// using BLAS
	Blob<Dtype> batch_sum_multiplier_;
	Blob<Dtype> num_by_chans_;
	Blob<Dtype> spatial_sum_multiplier_;
};

}  // namespace caffe