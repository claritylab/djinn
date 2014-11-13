
#pragma once

#include <thread>

#include "../../gen-cpp2/App.h"
#include "../../gen-cpp2/Dnn.h"
#include "../../gen-cpp2/dnn_types.h"
#include "caffe/caffe.hpp"

#include "nnet/nnet-nnet.h"                                                                                                           
#include "nnet/nnet-pdf-prior.h"                                                                                                      
#include "nnet/nnet-rbm.h"                                                                                                            
#include "base/kaldi-common.h"                                                                                                        
#include "util/common-utils.h"                                                                                                        

struct dim_t {
  int input_len;
  int n_in;
  int c_in;
  int h_in;
  int w_in;
};

class AppClientHandler : public facebook::windtunnel::treadmill::services::dnn::AppSvIf {
 public:
  AppClientHandler() { }
  AppClientHandler(std::string hostname, int port) {
    // Let's just load all the data for all the apps <---
    // IMC
    espresso = new caffe::Net<double>("input/imc-inputnet.prototxt");
    img_blobs = espresso->ForwardPrefilled(&loss);

    dim_t img_dim;
    img_dim.input_len = img_blobs[0]->count();
    data["imc"]= (double*)malloc(img_dim.input_len * sizeof(double));
    for(int i = 0; i < img_dim.input_len; ++i) {
      data["imc"][i] = img_blobs[0]->cpu_data()[i];
    }
     
    //TODO change?
    img_dim.n_in = 1; img_dim.c_in = 3; img_dim.w_in = 227; img_dim.h_in = 227;
    dimensions["imc"] = img_dim; 
   
    // ASR
    // Initilaization happens in future_asr() 
    
    // DIG
    // FACE
    // POS
    // NER
    // CHK
    // SRL
    
    hostname_ = hostname;
    port_ = port;
  }



  folly::wangle::Future<
    std::unique_ptr<
      facebook::windtunnel::treadmill::services::dnn::AppResult> > future_asr();
  folly::wangle::Future<
    std::unique_ptr<
      facebook::windtunnel::treadmill::services::dnn::AppResult> > future_imc();

  // folly::wangle::Future<
  //   std::unique_ptr<
  //     facebook::windtunnel::treadmill::services::dnn::AppResult> > future_pos();

 private:

  // hold app data
  std::map<std::string, double*> data;
  // copy the number of elts
  std::map<std::string, dim_t> dimensions;

  double loss;
  std::string hostname_;
  int port_;
  std::shared_ptr<facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient> client_;

  // Application specific stuff
  // IMC
  caffe::Net<double>* espresso;
  std::vector<caffe::Blob<double>* > img_blobs;

  // ASR
  dim_t asr_feat_dim;
  double* asr_feat;
  kaldi::nnet1::Nnet nnet_transf;

  // DIG
  // FACE
  // POS
  // NER
  // CHK
  // SRL
};
