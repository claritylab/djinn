
#pragma once

#include "../gen-cpp2/Dnn.h"

namespace facebook {
namespace windtunnel {
namespace treadmill {
namespace services {
namespace dnn{

class DnnHandler : public DnnSvAsyncIf {
  public:
    DnnHandler();

 
    folly::wangle::Future<Result> future_fwd(Work input);
};

} // namespace sleep
} // namespace services
} // namespace treadmill
} // namespace windtunnel
} // namespace facebook
