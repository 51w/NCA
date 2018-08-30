#pragma once
#include <string>

#include "blob.hpp"
#include "util/math_functions.hpp"
#include "proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Filler {
public:
	explicit Filler(const FillerParameter& param) : filler_param_(param) {}
	virtual ~Filler() {}
	virtual void Fill(Blob<Dtype>* blob) = 0;
protected:
	FillerParameter filler_param_;
};  // class Filler


template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
public:
	explicit ConstantFiller(const FillerParameter& param)
		: Filler<Dtype>(param) {}
	virtual void Fill(Blob<Dtype>* blob) {
		Dtype* data = blob->mutable_cpu_data();
		const int count = blob->count();
		const Dtype value = this->filler_param_.value();
		CHECK(count);
		for (int i = 0; i < count; ++i) {
			data[i] = value;
		}
		CHECK_EQ(this->filler_param_.sparse(), -1)
			<< "Sparsity not supported by this Filler.";
	}
};

template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
	const std::string& type = param.type();
	if (type == "constant") {
		return new ConstantFiller<Dtype>(param);
	}
	else if (type == "gaussian") {
		return new ConstantFiller<Dtype>(param);
	}
	else if (type == "positive_unitball") {
		return new ConstantFiller<Dtype>(param);
	}
	else if (type == "uniform") {
		return new ConstantFiller<Dtype>(param);
	}
	else if (type == "xavier") {
		return new ConstantFiller<Dtype>(param);
	}
	else if (type == "msra") {
		return new ConstantFiller<Dtype>(param);
	}
	else if (type == "bilinear") {
		return new ConstantFiller<Dtype>(param);
	}
	else {
		CHECK(false) << "Unknown filler name: " << param.type();
	}
	return (Filler<Dtype>*)(NULL);
}

}