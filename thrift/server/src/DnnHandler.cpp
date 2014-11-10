#include <unistd.h>
#include <string>
#include <map>

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

Future<std::unique_ptr<ServerResult> > DnnHandler::future_fwd(Work input) {
    folly::MoveWrapper<Promise<std::unique_ptr<ServerResult>>> promise;
    auto future = promise->getFuture();

    this->getEventBase()->runInEventBaseThread(
      [promise, this, input]() mutable {
        std::cout << "Task " << input.op << " forward pass.\n";

        if(init_pass) {
          // set caffe status and gpu
          Caffe::set_phase(Caffe::TEST);
          if(gpu) {
            Caffe::set_mode(Caffe::GPU);
            Caffe::SetDevice(0);
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

        vector<Blob<double>* > in_blobs = nets[input.op]->input_blobs();

        // Reshape the input dimension if neccessary
        int n_in = in_blobs[0]->num();
        int c_in = in_blobs[0]->channels();
        int w_in = in_blobs[0]->width();
        int h_in = in_blobs[0]->height();
                                                            
        if(input.n_in != n_in ||
           input.c_in != c_in ||
           input.w_in != w_in ||
           input.h_in != h_in) {
          // printf("Reshaping input and output dims to num:%d channel:%d width:%d height:%d\n",
          //           input.n_in. input.c_in, input.w_in, input.h_in);
          in_blobs[0]->Reshape(input.n_in, input.c_in, input.w_in, input.h_in);
          nets[input.op]->output_blobs()[0]->Reshape(input.n_in, input.c_in, input.w_in, input.h_in);
        }

        double* in = &input.data[0];
        in_blobs[0]->set_cpu_data(in);
        double loss;

        struct timeval start_time, end_time, diff_time;
        gettimeofday(&start_time, nullptr);

        vector<Blob<double>* > out_blobs = nets[input.op]->ForwardPrefilled(&loss);
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
