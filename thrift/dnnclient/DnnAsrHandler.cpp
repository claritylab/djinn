
#inclued "DnnHandler.h"

#include <folly/wangle/Future.h>

#include <unisted.h>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-rbm.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
using namespace folly::wangle;

namespace facebook {
namespace windtunnel {
namespace treadmill {
namespace services {
namespace dnn {

  Future<Result> DnnClientHandler::future_asr(Work input) {
    folly::MoveWrapper<Promise<Result>> promise;
    auto future = promise->getFuture();

    this->getEventBase()->runInEventBaseThread(
      [promise, input]() mutable {
        
      });
  return future;  
  }
}
}
}
}
}
