#ifndef DNN_H
#define DNN_H

#include "boost/program_options.hpp" 
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

namespace po = boost::program_options;

#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

// returns top layer dimension
Net<float>* SERVICE_init(po::variables_map& vm);

void SERVICE_fwd(float *in, int in_size, float *out, int out_size, Net<float>* net);

void SERVICE_close();

#endif
