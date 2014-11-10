
#pragma once

#include "../../gen-cpp2/App.h"
#include "../../gen-cpp2/Dnn.h"
#include "../../gen-cpp2/dnn_types.h"

class AppClientHandler : public facebook::windtunnel::treadmill::services::dnn::AppSvIf {
 public:
  AppClientHandler() { }
 
  /*
  folly::wangle::Future<
    std::unique_ptr<
      facebook::windtunnel::treadmill::services::dnn::AppResult> > future_asr();
  */
  folly::wangle::Future<
    std::unique_ptr<
      facebook::windtunnel::treadmill::services::dnn::AppResult> > future_imc();

 private:
  std::string hostname = "localhost";
  int port = 8081;
  //facebook::windtunnel::treadmill::services::dnn::Work work;
  std::unique_ptr<facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient> client;
};
