#pragma once
#include <string>
#include <vector>

#include "blob.hpp"
#include "common.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"
#include "io.hpp"

namespace caffe {

template <typename Dtype>
class Net {
public:
	explicit Net(const NetParameter& param);
	explicit Net(const string& param_file);
	virtual ~Net() {}

	void Init(const NetParameter& param);


	// mini-caffe
	void Forward();

	void Reshape();

	void CopyTrainedLayersFrom(const NetParameter& param);
	void CopyTrainedLayersFrom(const string trained_filename);

	inline const string& name() const { return name_; }
	inline const vector<string>& layer_names() const { return layer_names_; }
	inline const vector<string>& blob_names() const { return blob_names_; }
	inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
		return blobs_;
	}
	inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
		return layers_;
	}
	inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
		return bottom_vecs_;
	}
	inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
		return top_vecs_;
	}

	inline const vector<int> & top_ids(int i) const {
		CHECK_GE(i, 0) << "Invalid layer id";
		CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
		return top_id_vecs_[i];
	}
	inline const vector<int> & bottom_ids(int i) const {
		CHECK_GE(i, 0) << "Invalid layer id";
		CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
		return bottom_id_vecs_[i];
	}

	inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
		return params_;
	}


	bool has_blob(const string& blob_name) const;
	const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
	bool has_layer(const string& layer_name) const;
	const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;


//protected:
	void AppendTop(const NetParameter& param, const int layer_id,
		const int top_id, set<string>* available_blobs,
		map<string, int>* blob_name_to_idx);
	
	int AppendBottom(const NetParameter& param, const int layer_id,
		const int bottom_id, set<string>* available_blobs,
		map<string, int>* blob_name_to_idx);
	
	void AppendParam(const NetParameter& param, const int layer_id,
		const int param_id);


	string name_;
	vector<string> layer_names_;
	vector<shared_ptr<Layer<Dtype> > > layers_;
	map<string, int> layer_names_index_;

	//Blob
	vector<shared_ptr<Blob<Dtype> > > blobs_;
	vector<string> blob_names_;
	map<string, int> blob_names_index_;

	//HOST data
	vector<vector<Blob<Dtype>*> > bottom_vecs_;
	vector<vector<int> > bottom_id_vecs_;
	vector<vector<Blob<Dtype>*> > top_vecs_;
	vector<vector<int> > top_id_vecs_;

	vector<vector<int> > param_id_vecs_;
	vector<shared_ptr<Blob<Dtype> > > params_;

	DISABLE_COPY_AND_ASSIGN(Net);
};

}
