#ifndef TONIC_H
#define TONIC_H

#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;

// largest len: FACE + \0
#define MAX_REQ_SIZE 5

struct TonicPayload {
  // req name
  char req_name[MAX_REQ_SIZE];
  // size of req array
  int req_len;
  // size of image
  int img_size;
  // number of images to send (1 or batched)
  int num_imgs;
  // data
  float *data;
};

struct TonicSuiteApp {
  // Tonic App information
  // task
  std::string task;
  // network config
  std::string network;
  // pretrained weights
  std::string weights;
  // file with inputs
  std::string input;
  // use GPU?
  bool gpu;

  // internal net
  Net<float> *net;

  // use DjiNN service
  bool djinn;
  // hostname to send to
  std::string hostname;
  // port
  int portno;
  // socket descriptor
  int socketfd;

  // data to send to DjiNN service
  TonicPayload pl;

};

void reshape(Net<float> *net, int input_size);

#endif
