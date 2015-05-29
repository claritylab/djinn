#include <stdio.h>
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

using namespace std;

int translate_kaldi_model(char* weight_file_name, Net<float>* net, bool dump);
