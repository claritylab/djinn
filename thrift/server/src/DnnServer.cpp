#include <string>
#include <memory>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "thrift/lib/cpp2/server/ThriftServer.h"

#include "DnnHandler.h"
#include "caffe/caffe.hpp"

using namespace apache::thrift;
using namespace apache::thrift::async;
using namespace facebook::windtunnel::treadmill::services::dnn;
using namespace gflags;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

DEFINE_bool(gpu, true, "gpu flag");
DEFINE_int32(gpuid, 0, "gpu id");
DEFINE_bool(debug, false, "debug");
DEFINE_int32(port, 8000, "server port (default: 8000)");
DEFINE_int32(num_workers, 4, "Number of workers (default: 4)");

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  ParseCommandLineFlags(&argc, &argv, true);

  auto handler = std::make_shared<DnnHandler>(FLAGS_gpu, FLAGS_gpuid);
  auto server = folly::make_unique<ThriftServer>();

  server->setPort(FLAGS_port);
  server->setNWorkerThreads(FLAGS_num_workers);
  server->setInterface(std::move(handler));

  LOG(INFO) << "Listening on: " << FLAGS_port;
  if(FLAGS_gpu) LOG(INFO) << "Using GPU: " << FLAGS_gpuid;
  LOG(INFO) << "Workers: " << FLAGS_num_workers;

  server->serve();

  return 0;
}
