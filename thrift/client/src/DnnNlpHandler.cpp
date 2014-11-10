#include <unistd.h>
#include <folly/wangle/Future.h>
#include "DnnClientHandler.h"
#include "caffe/caffe.hpp"
// #include "SENNA_utils.h"
// #include "SENNA_Hash.h"
// #include "SENNA_Tokenizer.h"
// #include "SENNA_POS.h"
// #include "SENNA_CHK.h"
// #include "SENNA_NER.h"
// #include "SENNA_VBS.h"
// #include "SENNA_PT0.h"
// #include "SENNA_SRL.h"

using namespace folly::wangle;

namespace facebook {
namespace windtunnel {
namespace treadmill {
namespace services {
namespace dnn {

  Future<Result> DnnClientHandler::future_nlp(Work input) {
    folly::MoveWrapper<Promise<AppResult>> promise;
    auto future = promise->getFuture();

    this->getEventBase()->runInEventBaseThread(
      [promise, input]() mutable {
        
        int network_data_size = input.data.size() * sizeof(float);

        TEventBase event_base;

        std::shared_ptr<TAsyncSocket> socket(
          TAsyncSocket::newSocket(&event_base, "localhost", 8080));

        DnnAsyncClient client(
          std::unique_ptr<HeaderClientChannel, TDelayDestruction::Destructor>(
                   new HeaderClientChannel(socket)));

        gettimeofday(&comm_start_time, nullptr);
        if(FLAGS_task == "pos")
            pos_labels = SENNA_POS_forward(pos,
                  tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, client, input);

        gettimeofday(&app_end_time, nullptr);
        timersub(&app_end_time, &app_start_time, &app_diff_time);

        AppResult ret;
        ret.app_time = app_diff_time.tv_sec * 1000 + app_diff_time.tv_usec/1000; 
        ret.comm_time = (comm_diff_time.tv_sec * 1000 + comm_diff_time.tv_usec/1000)
                - service_result.fwd_time;
        ret.fwd_time = service_result.fwd_time;
        ret.comm_data_size = network_data_size;
        
        promise->setValue(std::move(ret));

      });

  return future;  
}

}
}
}
}
}
