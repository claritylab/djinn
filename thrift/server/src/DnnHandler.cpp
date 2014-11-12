#include <unistd.h>
#include <string>
#include <map>
#include <glog/logging.h>

#include "DnnHandler.h"
#include "caffe/caffe.hpp"

#include <folly/wangle/Future.h>

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

DnnHandler::DnnHandler(bool gpu, int gpuid)
{
  // set caffe status and gpu
  Caffe::set_phase(Caffe::TEST);
  if(gpu) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpuid);
  }else
    Caffe::set_mode(Caffe::CPU);

  std::ifstream file ("nets.txt");
  std::string net;
  while(file >> net)
  {
    Net<double>* temp = new Net<double>(net);
    const std::string name = temp->name();
    nets[name] = temp;
    std::string weights = "weights/" + name + ".caffemodel";
    nets[name]->CopyTrainedLayersFrom(weights);
  }
  init_pass = false;
}

folly::wangle::Future<std::unique_ptr<ServerResult> > DnnHandler::future_fwd(
  std::unique_ptr<Work> input) {
    folly::MoveWrapper<Promise<std::unique_ptr<ServerResult>>> promise;
    auto future = promise->getFuture();

    Work p_work = *input;
    this->getEventBase()->runInEventBaseThread(
      [this, promise, p_work]() mutable {
        // LOG(INFO) << "Task " << p_work.op << " forward pass.";
        // set caffe status and gpu

        vector<Blob<double>* > in_blobs = nets[p_work.op]->input_blobs();

        // Reshape the p_work dimension if neccessary
        int n_in = in_blobs[0]->num();
        int c_in = in_blobs[0]->channels();
        int w_in = in_blobs[0]->width();
        int h_in = in_blobs[0]->height();

        if(p_work.n_in != n_in ||
           p_work.c_in != c_in ||
           p_work.w_in != w_in ||
           p_work.h_in != h_in) {
          printf("Reshaping p_work and output dims to num:%d channel:%d width:%d height:%d\n",
                    p_work.n_in, p_work.c_in, p_work.w_in, p_work.h_in);
          in_blobs[0]->Reshape(p_work.n_in, p_work.c_in, p_work.w_in, p_work.h_in);
          nets[p_work.op]->output_blobs()[0]->Reshape(p_work.n_in, p_work.c_in, p_work.w_in, p_work.h_in);
        }

        double* in = &p_work.data[0];
        in_blobs[0]->set_cpu_data(in);
        double loss;

        struct timeval start_time, end_time, diff_time;
        gettimeofday(&start_time, nullptr);

        vector<Blob<double>* > out_blobs = nets[p_work.op]->ForwardPrefilled(&loss);
        gettimeofday(&end_time, nullptr);

        timersub(&end_time, &start_time, &diff_time);

        std::unique_ptr<ServerResult> temp_result(new ServerResult());
        temp_result->time_ms = diff_time.tv_sec * 1000 + diff_time.tv_usec / 1000;

        for(int i = 0; i < out_blobs[0]->count(); ++i)
          temp_result->data.push_back(out_blobs[0]->cpu_data()[i]);

        // Set promise value
        promise->setValue(std::move(temp_result));
      });
  return future;  
  }
