
#inclued "DnnHandler.h"
#include "caffe/caffe.hpp"

#include <folly/wangle/Future.h>

#include <unisted.h>

using namespace folly::wangle;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


namespace facebook {
namespace windtunnel {
namespace treadmill {
namespace services {
namespace dnn {

  DnnHander::DnnHandler() {
    ifstream file (FLAGS_net_list.c_str());

    string net;
    while(file >> net){
      Net<double>* temp = new Net<double>(net);
      const string name = temp->name();
      nets[name] = temp;
      string weights = FLAGS_net_weights + name + ".caffemodel";
      nets[name]->CopyTrainedLayersFrom(weights);
      nets[name]->set_debug_info(FLAGS_debug);
    }
  }

  Future<Result> DnnHandler::future_fwd(Work input) {
    folly::MoveWrapper<Promise<Result>> promise;
    auto future = promise->getFuture();

    this->getEventBase()->runInEventBaseThread(
      [promise, input]() mutable {
        cout << "Task " << input.op << " forward pass.\n";
        
        Blob<double>* in_blobs = nets[input.op]->input_blobs()[0];

        // Reshape the input dimension if neccessary
        int n_in = in_blobs->num();
        int c_in = in_blobs->channels();
        int w_in = in_blobs->width();
        int h_in = in_blobs->height();
        int in_elts = in_blobs->count();
                                                            
        if(input.n_in != n_in ||
                input.c_in != c_in ||
                input.w_in != w_in ||
                input.h_in != h_in){
          printf("Reshaping input and output dims to num:%d channel:%d width:%d height:%d\n",
                    input.n_in. input.c_in, input.w_in, input.h_in);
          in_blobs->Reshapw(input.n_in, input.c_in, input.w_in. input.h_in);
          nets[input.op]->output_blobs()[0]->Reshape(input.n_in, input.c_in, input.w_in, input.h_in);
        }

        double* in = &input.data[0];
        in_blobs[0]->set_cpu_data(in);
        double loss;

        struct timeval start_time, end_time. diff_time;
        gettimeofday(&start_time, nullptr);

        vector<Blob<double>* > out_blobs = nets[input.op]->ForwardPrefilled(&loss);
        gettimeofday(&end_time, nullptr);

        timersub(&end)time, &start_time, &diff_time);

        Result temp_result;
        temp_result.time_ms = diff_time.tv_sec * 1000 + diff_time.tv_usec / 1000;

        for(int i = 0; i < out_blobs[0]->count(); ++i){
          temp_result.data.push_back(out_blobs[0]->cpu_data()[i]);
        }
       
        // Set promise value
        promise->setValue(std::move(temp_result));
      });
  return future;  
  }
}
}
}
}
}
