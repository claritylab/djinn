
#pragma once

#include <thread>

#include "../../gen-cpp2/App.h"
#include "../../gen-cpp2/Dnn.h"
#include "../../gen-cpp2/dnn_types.h"
#include "caffe/caffe.hpp"

class AppClientHandler : public facebook::windtunnel::treadmill::services::dnn::AppSvIf {
 public:
  AppClientHandler() { }
  AppClientHandler(std::string hostname, int port) {
    // Let's just load all the data for all the apps <---
    espresso = new caffe::Net<double>("input/imc-inputnet.prototxt");
    img_blobs = espresso->ForwardPrefilled(&loss);

    input_len = img_blobs[0]->count();
    data = (double*)malloc(input_len * sizeof(double));
    for(int i = 0; i < input_len; ++i) {
      data[i] = img_blobs[0]->cpu_data()[i];
    }

    //TODO change?
    n_in = 1; c_in = 3; w_in = 227; h_in = 227;
    hostname_ = hostname;
    port_ = port;
  
  }

  /*
  folly::wangle::Future<
    std::unique_ptr<
      facebook::windtunnel::treadmill::services::dnn::AppResult> > future_asr();
  */
  folly::wangle::Future<
    std::unique_ptr<
      facebook::windtunnel::treadmill::services::dnn::AppResult> > future_imc();

  // folly::wangle::Future<
  //   std::unique_ptr<
  //     facebook::windtunnel::treadmill::services::dnn::AppResult> > future_pos();

 private:
  caffe::Net<double>* espresso;
  std::vector<caffe::Blob<double>* > img_blobs;
  // hold app data
  double *data;
  // copy the number of elts
  int input_len;
  int n_in;
  int c_in;
  int h_in;
  int w_in;
  double loss;
  std::string hostname_;
  int port_;
  std::shared_ptr<facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient> client_;
};
