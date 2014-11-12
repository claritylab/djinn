
#pragma once

#include "../../gen-cpp2/Dnn.h"
#include "caffe/caffe.hpp"

class DnnHandler : public facebook::windtunnel::treadmill::services::dnn::DnnSvIf {
 public:
  DnnHandler();
  // DnnHandler(bool gpu, int gpuid);

  folly::wangle::Future<
    std::unique_ptr<facebook::windtunnel::treadmill::services::dnn::ServerResult> >
      future_fwd(std::unique_ptr<facebook::windtunnel::treadmill::services::dnn::Work> input);

 private:
  std::map<std::string, caffe::Net<double>* > nets;
  bool init_pass = true;
  bool gpu = true;
};
