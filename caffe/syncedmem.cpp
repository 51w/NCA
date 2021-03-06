#include <cstring>
#include "syncedmem.hpp"

namespace caffe {

SyncedMemory::SyncedMemory()
	: cpu_ptr_(NULL), size_(0), head_(UNINITIALIZED), own_cpu_data_(false)
{}

SyncedMemory::SyncedMemory(size_t size)
	: cpu_ptr_(NULL), size_(size), head_(UNINITIALIZED), own_cpu_data_(false)
{}

SyncedMemory::~SyncedMemory() 
{
	if (cpu_ptr_ && own_cpu_data_)
	{  CaffeFreeHost(cpu_ptr_); }
}

inline void SyncedMemory::to_cpu()
{
	switch (head_) 
	{
	case UNINITIALIZED:
		CaffeMallocHost(&cpu_ptr_, size_);
		//memset(size_, 0, cpu_ptr_);
		memset(cpu_ptr_, 0, size_);
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
		break;
	case HEAD_AT_CPU:
		break;
	}
}

const void* SyncedMemory::cpu_data() 
{
	to_cpu();
	return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) 
{
	CHECK(data);
	if (own_cpu_data_) {
		CaffeFreeHost(cpu_ptr_);
	}
	cpu_ptr_ = data;
	head_ = HEAD_AT_CPU;
	own_cpu_data_ = false;
}

void* SyncedMemory::mutable_cpu_data() 
{
	to_cpu();
	head_ = HEAD_AT_CPU;
	return cpu_ptr_;
}

}