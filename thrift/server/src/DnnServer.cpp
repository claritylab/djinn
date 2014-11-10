#include <string>
#include <memory>
#include <gflags/gflags.h>

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

DEFINE_bool(gpu, false, "gpu flag");
DEFINE_bool(debug, false, "debug");
DEFINE_int32(port, 8080, "server port (default: 8080)");
DEFINE_int32(num_workers, 8, "Number of workers (default: 4)");

int main(int argc, char* argv[])
{
  ParseCommandLineFlags(&argc, &argv, true);

  auto handler = std::make_shared<DnnHandler>();
  auto server = folly::make_unique<ThriftServer>();

  server->setPort(FLAGS_port);
  server->setNWorkerThreads(FLAGS_num_workers);
  server->setInterface(std::move(handler));

  server->serve();

  return 0;
}
