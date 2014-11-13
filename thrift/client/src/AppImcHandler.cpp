#include <unistd.h>
#include <sys/time.h>
#include <folly/wangle/Future.h>
#include <glog/logging.h>

#include "AppClientHandler.h"
#include "caffe/caffe.hpp"

using namespace folly::wangle;
using namespace apache::thrift;
using namespace apache::thrift::async;
using namespace facebook::windtunnel::treadmill::services::dnn;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

Future<std::unique_ptr<AppResult> > AppClientHandler::future_imc() {

  if (client_ == nullptr) {
    std::shared_ptr<apache::thrift::async::TAsyncSocket> socket(
        apache::thrift::async::TAsyncSocket::newSocket(this->getEventBase(),
                                                             hostname_,
                                                             port_));

    client_ = std::shared_ptr<
                  facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient>(
                    new facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient(
                      std::unique_ptr<apache::thrift::HeaderClientChannel,
                      apache::thrift::TDelayedDestruction::Destructor>(
                          new apache::thrift::HeaderClientChannel(socket)
                        )
                      )
                    );
  }

  folly::MoveWrapper<Promise<std::unique_ptr<AppResult> > > promise;
  auto future = promise->getFuture();

  this->getEventBase()->runInEventBaseThread(
    [this, promise]() mutable {
      
    struct timeval app_start_time, app_end_time, app_diff_time;
    struct timeval comm_start_time, comm_end_time, comm_diff_time;
    Work work;

    // start of application
    gettimeofday(&app_start_time, nullptr);

    // prepare data
    work.op = "imc";
    dim_t dim = dimensions["imc"];
    work.n_in = dim.n_in; work.c_in = dim.c_in; work.w_in = dim.w_in; work.h_in = dim.h_in;

    for(int i = 0; i < dim.input_len; ++i)
      work.data.push_back(data["imc"][i]);

    int network_data_size = work.data.size() * sizeof(float);

    gettimeofday(&comm_start_time, nullptr);

    auto f = client_->future_fwd(work);

    f.then(
      [promise, app_start_time, app_end_time, app_diff_time,
       comm_start_time, comm_end_time, comm_diff_time,
       network_data_size](folly::wangle::Try<ServerResult>&& t) mutable {
      // end of application
      gettimeofday(&app_end_time, nullptr);
      timersub(&app_end_time, &app_start_time, &app_diff_time);

      std::unique_ptr<AppResult> ret(new AppResult());;
      ret->app_time = app_diff_time.tv_sec * 1000 + app_diff_time.tv_usec / 1000; 

      ret->comm_time = (comm_diff_time.tv_sec * 1000 + comm_diff_time.tv_usec / 1000)
        - t.value().time_ms;
      ret->fwd_time = t.value().time_ms;
      ret->comm_data_size = network_data_size;

      for(unsigned int j = 0; j < t.value().data.size(); ++j)
        LOG(INFO) << "Image: " << j << " class: " << t.value().data[j];

      promise->setValue(std::move(ret));
    });
    return f;
  });

  return future;
}
