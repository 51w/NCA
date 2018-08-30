#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "common.hpp"
#include "blob.hpp"
#include "layer_factory.hpp"
#include "proto/caffe.pb.h"
#include "util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class Layer {
public:
	explicit Layer(const LayerParameter& param)
		: layer_param_(param) {
		// Set phase and copy blobs (if there are any).
		if (layer_param_.blobs_size() > 0) {
			blobs_.resize(layer_param_.blobs_size());
			for (int i = 0; i < layer_param_.blobs_size(); ++i) {
				blobs_[i].reset(new Blob<Dtype>());
				blobs_[i]->FromProto(layer_param_.blobs(i));
			}
		}
	}
	virtual ~Layer() {}

	void SetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CheckBlobCounts(bottom, top);
		LayerSetUp(bottom, top);
		Reshape(bottom, top);
	}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {}

	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) = 0;

	inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);




	vector<shared_ptr<Blob<Dtype> > >& blobs() {
		return blobs_;
	}

	const LayerParameter& layer_param() const { return layer_param_; }

	virtual inline const char* type() const { return ""; }


	virtual inline int ExactNumBottomBlobs() const { return -1; }
	virtual inline int MinBottomBlobs() const { return -1; }
	virtual inline int MaxBottomBlobs() const { return -1; }
	virtual inline int ExactNumTopBlobs() const { return -1; }
	virtual inline int MinTopBlobs() const { return -1; }
	virtual inline int MaxTopBlobs() const { return -1; }
	virtual inline bool EqualNumBottomTopBlobs() const { return false; }
	virtual inline bool AutoTopBlobs() const { return false; }

protected:
	/** The protobuf that stores the layer parameters */
	LayerParameter layer_param_;
	vector<shared_ptr<Blob<Dtype> > > blobs_;

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) = 0;

	virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (ExactNumBottomBlobs() >= 0) {
			CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
				<< type() << " Layer takes " << ExactNumBottomBlobs()
				<< " bottom blob(s) as input.";
		}
		if (MinBottomBlobs() >= 0) {
			CHECK_LE(MinBottomBlobs(), bottom.size())
				<< type() << " Layer takes at least " << MinBottomBlobs()
				<< " bottom blob(s) as input.";
		}
		if (MaxBottomBlobs() >= 0) {
			CHECK_GE(MaxBottomBlobs(), bottom.size())
				<< type() << " Layer takes at most " << MaxBottomBlobs()
				<< " bottom blob(s) as input.";
		}
		if (ExactNumTopBlobs() >= 0) {
			CHECK_EQ(ExactNumTopBlobs(), top.size())
				<< type() << " Layer produces " << ExactNumTopBlobs()
				<< " top blob(s) as output.";
		}
		if (MinTopBlobs() >= 0) {
			CHECK_LE(MinTopBlobs(), top.size())
				<< type() << " Layer produces at least " << MinTopBlobs()
				<< " top blob(s) as output.";
		}
		if (MaxTopBlobs() >= 0) {
			CHECK_GE(MaxTopBlobs(), top.size())
				<< type() << " Layer produces at most " << MaxTopBlobs()
				<< " top blob(s) as output.";
		}
		if (EqualNumBottomTopBlobs()) {
			CHECK_EQ(bottom.size(), top.size())
				<< type() << " Layer produces one top blob as output for each "
				<< "bottom blob input.";
		}
	}


private:
	DISABLE_COPY_AND_ASSIGN(Layer);
};


template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	Dtype loss = 0;
	Forward_cpu(bottom, top);
	return loss;
}

}