#pragma once
#include <cstdlib>

#include "common.hpp"

namespace caffe {

inline void CaffeMallocHost(void** ptr, size_t size) 
{
	*ptr = malloc(size);
	CHECK(*ptr) << "host allocation of size " << size << " failed";
	// if (!(*ptr)) {
		// fprintf(stderr, "host allocation of size %d failed", size);
		// abort();
	// }
}

inline void CaffeFreeHost(void* ptr)
{
	free(ptr);
}


class SyncedMemory {
public:
	SyncedMemory();
	explicit SyncedMemory(size_t size);
	~SyncedMemory();
	const void* cpu_data();
	void set_cpu_data(void* data);

	void* mutable_cpu_data();
	enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU };
	SyncedHead head() const { return head_; }
	size_t size() const { return size_; }

private:
	void to_cpu();
	void* cpu_ptr_;
	size_t size_;
	SyncedHead head_;
	bool own_cpu_data_;

	DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};

}