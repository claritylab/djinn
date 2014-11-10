#include <unistd.h>
#include <sys/time.h>
#include <folly/wangle/Future.h>

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
  folly::MoveWrapper<Promise<std::unique_ptr<AppResult>>> promise;
  auto future = promise->getFuture();

  this->getEventBase()->runInEventBaseThread(
    [promise, this]() mutable {
      
    printf("received\n");
    struct timeval app_start_time, app_end_time, app_diff_time;
    struct timeval comm_start_time, comm_end_time, comm_diff_time;
    Work work;

      // start of application
      gettimeofday(&app_start_time, nullptr);
      if(client == nullptr) {
        printf("init phase\n");
        work.op = "imc";
        Net<double>* espresso = new Net<double>("input/imc-inputnet.prototxt");
      printf("1\n");
        // Blob<double>* in_blobs = espresso->input_blobs()[0];
        //
        // work.n_in = in_blobs->num();
        // work.c_in = in_blobs->channels();
        // work.w_in = in_blobs->width();
        // work.h_in = in_blobs->height();
        work.n_in = 1;
        work.c_in = 3;
        work.w_in = 227;
        work.h_in = 227;

      printf("1\n");
        double loss;
        vector<Blob<double>* > img_blobs = espresso->ForwardPrefilled(&loss);
      printf("2\n");
        for(int i = 0; i <img_blobs[0]->count(); ++i)
          work.data.push_back(img_blobs[0]->cpu_data()[i]);
      printf("3\n");

        apache::thrift::async::TEventBase event_base;
        std::shared_ptr<apache::thrift::async::TAsyncSocket> socket(
          apache::thrift::async::TAsyncSocket::newSocket(&event_base, hostname, port));
        client = std::unique_ptr<
                      facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient>(
                        new facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient(
                          std::unique_ptr<apache::thrift::HeaderClientChannel,
                          apache::thrift::TDelayedDestruction::Destructor>(
                              new apache::thrift::HeaderClientChannel(socket)
                            )
                          )
                        );

      }

      int network_data_size = work.data.size() * sizeof(float);
      printf("done phase\n");

      gettimeofday(&comm_start_time, nullptr);

      auto f = client->future_fwd(work).then(
        [promise, app_start_time, app_end_time, app_diff_time,
         comm_start_time, comm_end_time, comm_diff_time,
         network_data_size](folly::wangle::Try<std::unique_ptr<ServerResult>>&& t) mutable {

        printf("hey johann!\n");
        // end of application
        /*
        gettimeofday(&app_end_time, nullptr);
        timersub(&app_end_time, &app_start_time, &app_diff_time);

        std::unique_ptr<AppResult> ret(new AppResult());;
        ret->app_time = app_diff_time.tv_sec * 1000 + app_diff_time.tv_usec / 1000; 

        ret->comm_time = (comm_diff_time.tv_sec * 1000 + comm_diff_time.tv_usec / 1000)
          - t.value()->time_ms;
        ret->fwd_time = t.value()->time_ms;
        ret->comm_data_size = network_data_size;
        for(unsigned int j = 0; j < t.value()->data.size(); ++j)
          std::cout << "Image: " << j << " class: " << t.value()->data[j] << std::endl;
        */

        promise->setValue(std::move(ret));
      });
  });

  return future;
}
