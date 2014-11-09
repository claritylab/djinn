
#pragma once

#include "../gen-cpp2/Dnn.h"

namespace facebook {
namespace windtunnel {
namespace treadmill {
namespace services {
namespace dnn{

class DnnClientHandler : public DnnSvAsyncIf {
  public:
    DnnClientHandler();
 
    folly::wangle::Future<Result> future_asr(Work input);
    folly::wangle::Future<Result> future_imc(Work input);
};

} // namespace dnn
} // namespace services
} // namespace treadmill
} // namespace windtunnel
} // namespace facebook
