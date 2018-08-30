#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "common.hpp"
#include "proto/caffe.pb.h"
#include "syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {

template <typename Dtype>
class Blob {
public:
	Blob()
		: data_(), count_(0), capacity_(0) {}
	explicit Blob(const int num, const int channels, const int height,
		const int width);
	explicit Blob(const vector<int>& shape);

	void Reshape(const int num, const int channels, const int height,
		const int width);

	void Reshape(const vector<int>& shape);
	void Reshape(const BlobShape& shape);
	void ReshapeLike(const Blob& other);
	inline string shape_string() const 
	{
		ostringstream stream;
		for (int i = 0; i < shape_.size(); ++i) 
		{
			stream << shape_[i] << " ";
		}
		stream << "(" << count_ << ")";
		return stream.str();
	}
	inline const vector<int>& shape() const { return shape_; }


	//**************//
	inline int shape(int index) const {
		return shape_[CanonicalAxisIndex(index)];
	}
	inline int num_axes() const { return shape_.size(); }
	inline int count() const { return count_; }

	inline int count(int start_axis, int end_axis) const 
	{
		CHECK_LE(start_axis, end_axis);
		CHECK_GE(start_axis, 0);
		CHECK_GE(end_axis, 0);
		CHECK_LE(start_axis, num_axes());
		CHECK_LE(end_axis, num_axes());
		int count = 1;
		for (int i = start_axis; i < end_axis; ++i) 
		{
			count *= shape(i);
		}
		return count;
	}

	inline int count(int start_axis) const 
	{
		return count(start_axis, num_axes());
	}

	inline int CanonicalAxisIndex(int axis_index) const 
	{
		CHECK_GE(axis_index, -num_axes())
			<< "axis " << axis_index << " out of range for " << num_axes()
			<< "-D Blob with shape " << shape_string();
		CHECK_LT(axis_index, num_axes())
			<< "axis " << axis_index << " out of range for " << num_axes()
			<< "-D Blob with shape " << shape_string();
		if (axis_index < 0) {
			return axis_index + num_axes();
		}
		return axis_index;
	}


	// num channels height width
	inline int num() const { return LegacyShape(0); }
	inline int channels() const { return LegacyShape(1); }
	inline int height() const { return LegacyShape(2); }
	inline int width() const { return LegacyShape(3); }
	inline int LegacyShape(int index) const 
	{
		CHECK_LE(num_axes(), 4)
			<< "Cannot use legacy accessors on Blobs with > 4 axes.";
		CHECK_LT(index, 4);
		CHECK_GE(index, -4);
		if (index >= num_axes() || index < -num_axes()) 
		{
			// Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
			// indexing) -- this special case simulates the one-padding used to fill
			// extraneous axes of legacy blobs.
			return 1;
		}
		return shape(index);
	}

	inline int offset(const int n, const int c = 0, const int h = 0,
		const int w = 0) const 
	{
		CHECK_GE(n, 0);
		CHECK_LE(n, num());
		CHECK_GE(channels(), 0);
		CHECK_LE(c, channels());
		CHECK_GE(height(), 0);
		CHECK_LE(h, height());
		CHECK_GE(width(), 0);
		CHECK_LE(w, width());
		return ((n * channels() + c) * height() + h) * width() + w;
	}

	inline int offset(const vector<int>& indices) const 
	{
		CHECK_LE(indices.size(), num_axes());
		int offset = 0;
		for (int i = 0; i < num_axes(); ++i) 
		{
			offset *= shape(i);
			if (indices.size() > i) {
				CHECK_GE(indices[i], 0);
				CHECK_LT(indices[i], shape(i));
				offset += indices[i];
			}
		}
		return offset;
	}


	//************************//Copy from a source Blob.
	void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
		bool reshape = false);

	inline Dtype data_at(const int n, const int c, const int h,
		const int w) const 
	{
		return cpu_data()[offset(n, c, h, w)];
	}

	inline Dtype data_at(const vector<int>& index) const
	{
		return cpu_data()[offset(index)];
	}

	inline const shared_ptr<SyncedMemory>& data() const
	{
		CHECK(data_);
		return data_;
	}

	const Dtype* cpu_data() const;
	void set_cpu_data(Dtype* data);
	Dtype* mutable_cpu_data();

	void FromProto(const BlobProto& proto, bool reshape = true);

	void ShareData(const Blob& other);
	bool ShapeEquals(const BlobProto& other);

protected:
	shared_ptr<SyncedMemory> data_;
	shared_ptr<SyncedMemory> shape_data_;
	vector<int> shape_;
	int count_;
	int capacity_;

	DISABLE_COPY_AND_ASSIGN(Blob);
};

}