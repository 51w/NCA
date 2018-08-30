#pragma once
#include <vector>

#include "../layer.hpp"
#include "../common.hpp"

namespace caffe {

template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
public:
	explicit SplitLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Split"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	int count_;
};

}