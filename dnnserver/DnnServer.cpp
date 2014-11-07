#include <memory>
#include <gflags/gflags.h>

#include "thrift/lib/cpp2/server/ThriftServer.h"

#include "DnnHandler.h"
#include "caffe/caffe.hpp"

using namespace apache::thrift;
using namespace apache::thrift::async;

using namespace boost;

using namespace dnn;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_bool(gpu, false, "gpu flag");
DEFINE_bool(debug, false, "debug");
DEFINE_int32(port, 8080, "server port (default: 8080)");
DEFINE_string(gpu_mode, "pool", 
                "mode of gpu configuration (default: pool - pick which ever one is free.)")
DEFINE_string(net_list, "nets.txt", "file with net-configs");
DEFINE_string(net_weights, "weights/". "directory with weights");
DEFINE_string(num_workers, 4, "Number of workers (default: 4)");

int main(int argc, char* argv[]) {
        
  google::InitGoogleLogging(argv[0]);

  google::ParseCommandLineFlags(&argc, &argv, true);

  Caffe::set_phase(Caffe::TEST);
  
  if(FLAGS_gpu) {
    Caffe::set_mode(Caffe::GPU);
  }else
    Caffe::set_mode(Caffe::CPU);

  auto handler = std::make_shared<DnnHandler>();
  auto server = folly::make_unique<ThriftServer>();

  server->setPort(FLAGS_port);
  server->setNWorkerThreads(FLAGS_num_workers);
  server->setInterface(std::move(handler));

  server->serve();

  return 0;
}
