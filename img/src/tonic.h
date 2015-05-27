#ifndef TONIC_H
#define TONIC_H

#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;

struct TonicSuiteApp {
  // Tonic App information
  std::string task;
  std::string network;
  std::string model;
  std::string input;
  bool gpu;

  Net<float> *net;
  int img_size;
  int num_imgs;

  // DjiNN service
  bool djinn;
  std::string hostname;
  std::string portno;
  int socketfd;
};

void reshape(Net<float> *net, int input_size);

#endif
