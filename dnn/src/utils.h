// utils.h
// Yiping Kang (ypkang@umich.edu)
// 2014

#include <stdio.h>
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"
#include "SENNA_VBS.h"
#include "SENNA_PT0.h"
#include "SENNA_SRL.h"

#define DEBUG 0

using namespace std;


int translate_kaldi_model(char* weight_file_name, Net<float>* net, bool dump);
