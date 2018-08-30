#include <string>
#include <vector>

#include "net.hpp"
//#include "io.hpp"
#include "proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param)
{
	Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file)
{
	NetParameter param;
	CHECK(read_proto_from_text(param_file, &param))
		<< "Failed to parse NetParameter file: " << param_file;

	Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& param)
{
	name_ = param.name();
	map<string, int> blob_name_to_idx;
	set<string> available_blobs;
	// For each layer, set up its input and output
	bottom_vecs_.resize(param.layer_size());
	top_vecs_.resize(param.layer_size());
	bottom_id_vecs_.resize(param.layer_size());
	top_id_vecs_.resize(param.layer_size());
	param_id_vecs_.resize(param.layer_size());
	for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
		// Setup layer.
		const LayerParameter& layer_param = param.layer(layer_id);
		layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
		layer_names_.push_back(layer_param.name());
		// Figure out this layer's input and output
		const int num_bottom = layer_param.bottom_size();
		for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
			AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
		}
		const int num_top = layer_param.top_size();
		for (int top_id = 0; top_id < num_top; ++top_id) {
			AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
		}
		// After this layer is connected, set it up.
		layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
	}
	CHECK_EQ(std::string(layers_[0]->type()), std::string("Input"))
		<< "Network\'s first layer should be Input Layer.";

	for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
		blob_names_index_[blob_names_[blob_id]] = blob_id;
	}
	for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
		layer_names_index_[layer_names_[layer_id]] = layer_id;
	}

	//for (size_t blob_id = 0; blob_id < blobs_.size(); ++blob_id) {
	//	shared_ptr<Blob<Dtype> > blob_ptr;
	//	blob_ptr = blobs_[blob_id];
	//	blob_ptr->cpu_data();
	//}
}

template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
	const int top_id, set<string>* available_blobs,
	map<string, int>* blob_name_to_idx) 
{
	shared_ptr<LayerParameter> layer_param(
		new LayerParameter(param.layer(layer_id)));
	const string& blob_name = (layer_param->top_size() > top_id) ?
		layer_param->top(top_id) : "(automatic)";
	// Check if we are doing in-place computation
	if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
		blob_name == layer_param->bottom(top_id)) {
		// In-place computation
		top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
		top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
	}
	else if (blob_name_to_idx &&
		blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
		// If we are not doing in-place computation but have duplicated blobs,
		// raise an error.
		LOG(FATAL) << "Top blob '" << blob_name
			<< "' produced by multiple sources.";
	}
	else {
		// Normal output.
		shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
		const int blob_id = blobs_.size();
		blobs_.push_back(blob_pointer);
		blob_names_.push_back(blob_name);
		
		if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
		top_id_vecs_[layer_id].push_back(blob_id);
		top_vecs_[layer_id].push_back(blob_pointer.get());
	}
	if (available_blobs) { available_blobs->insert(blob_name); }
}

template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
	const int bottom_id, set<string>* available_blobs,
	map<string, int>* blob_name_to_idx) 
{
	const LayerParameter& layer_param = param.layer(layer_id);
	const string& blob_name = layer_param.bottom(bottom_id);
	if (available_blobs->find(blob_name) == available_blobs->end()) {
		LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
			<< layer_param.name() << "', bottom index " << bottom_id << ")";
	}
	const int blob_id = (*blob_name_to_idx)[blob_name];
	bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
	bottom_id_vecs_[layer_id].push_back(blob_id);
	
	return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
	const int param_id) 
{
	const LayerParameter& layer_param = layers_[layer_id]->layer_param();
	const int param_size = layer_param.param_size();
	string param_name =
		(param_size > param_id) ? layer_param.param(param_id).name() : "";
	
	const int net_param_id = params_.size();
	params_.push_back(layers_[layer_id]->blobs()[param_id]);
	param_id_vecs_[layer_id].push_back(net_param_id);
}

template <typename Dtype>
void Net<Dtype>::Forward() 
{
	for (int i = 0; i < layers_.size(); ++i) 
	{
		// if(strcmp(layer_names_[i].c_str(), "conv1") == 0)
		// {
			// LOG(INFO) << layer_names_[i];
			// LOG(INFO) << bottom_vecs_[i][0]->channels() << " " << bottom_vecs_[i][0]->height() << " "
			// << bottom_vecs_[i][0]->width();
			
			// FILE* pp1 = fopen("conv1.txt", "wb");
			// float *out1 = bottom_vecs_[i][0]->mutable_cpu_data();
			// for (int j=0; j < bottom_vecs_[i][0]->count(); j++)
				// fprintf(pp1, "%f  -- conv1\n", out1[j]);
			// fclose(pp1);
		// }
		
		layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
		
		// if(strcmp(layer_names_[i].c_str(), "conv1") == 0)
		// {
			// LOG(INFO) << layer_names_[i];
			// LOG(INFO) << top_vecs_[i][0]->channels() << " " << top_vecs_[i][0]->height() << " "
			// << top_vecs_[i][0]->width();
			
			// FILE* pp1 = fopen("conv2.txt", "wb");
			// float *out1 = top_vecs_[i][0]->mutable_cpu_data();
			// for (int j=0; j < top_vecs_[i][0]->count(); j++)
				// fprintf(pp1, "%f  -- conv2\n", out1[j]);
			// fclose(pp1);
		// }
	}
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
	for (int i = 0; i < layers_.size(); ++i) {
		layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
	}
}


template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
	int num_source_layers = param.layer_size();
	for (int i = 0; i < num_source_layers; ++i) {
		const LayerParameter& source_layer = param.layer(i);
		const string& source_layer_name = source_layer.name();
		int target_layer_id = 0;
		while (target_layer_id != layer_names_.size() &&
			layer_names_[target_layer_id] != source_layer_name) {
			++target_layer_id;
		}
		if (target_layer_id == layer_names_.size()) {
			//LOG(INFO) << "Ignoring source layer " << source_layer_name;
			continue;
		}
		//DLOG(INFO) << "Copying source layer " << source_layer_name;
		vector<shared_ptr<Blob<Dtype> > >& target_blobs =
			layers_[target_layer_id]->blobs();
		CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
			<< "Incompatible number of blobs for layer " << source_layer_name;
		for (int j = 0; j < target_blobs.size(); ++j) {
			if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
				Blob<Dtype> source_blob;
				const bool kReshape = true;
				source_blob.FromProto(source_layer.blobs(j), kReshape);
				LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
					<< source_layer_name << "'; shape mismatch.  Source param shape is "
					<< source_blob.shape_string() << "; target param shape is "
					<< target_blobs[j]->shape_string() << ". "
					<< "To learn this layer's parameters from scratch rather than "
					<< "copying from a saved net, rename the layer.";
			}
			const bool kReshape = false;
			target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
		}
	}
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) 
{
	NetParameter param;
	read_proto_from_binary(trained_filename, &param);
	CopyTrainedLayersFrom(param);
}



//
template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
	return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
	const string& blob_name) const {
	shared_ptr<Blob<Dtype> > blob_ptr;
	if (has_blob(blob_name)) {
		blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
	}
	else {
		blob_ptr.reset((Blob<Dtype>*)(NULL));
		LOG(WARNING) << "Unknown blob name " << blob_name;
	}
	return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
	return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
	const string& layer_name) const {
	shared_ptr<Layer<Dtype> > layer_ptr;
	if (has_layer(layer_name)) {
		layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
	}
	else {
		layer_ptr.reset((Layer<Dtype>*)(NULL));
		LOG(WARNING) << "Unknown layer name " << layer_name;
	}
	return layer_ptr;
}


INSTANTIATE_CLASS(Net);

}