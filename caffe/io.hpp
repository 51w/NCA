#pragma once
#include <fstream>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "proto/caffe.pb.h"


inline bool read_proto_from_text(std::string filepath, google::protobuf::Message* message)
{
	std::ifstream fs(filepath, std::ifstream::in);
	if (!fs.is_open())
	{
		return false;
	}

	google::protobuf::io::IstreamInputStream input(&fs);
	bool success = google::protobuf::TextFormat::Parse(&input, message);

	fs.close();

	return success;
}

inline bool read_proto_from_binary(std::string filepath, google::protobuf::Message* message)
{
	std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
	if (!fs.is_open())
	{
		return false;
	}

	google::protobuf::io::IstreamInputStream input(&fs);
	google::protobuf::io::CodedInputStream codedstr(&input);

	codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

	bool success = message->ParseFromCodedStream(&codedstr);

	fs.close();

	return success;
}
