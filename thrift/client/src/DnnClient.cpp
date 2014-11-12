#include <memory>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "thrift/lib/cpp2/server/ThriftServer.h"

#include "AppClientHandler.h"
#include "caffe/caffe.hpp"

using namespace apache::thrift;
using namespace apache::thrift::async;
using namespace facebook::windtunnel::treadmill::services::dnn;

using namespace boost;
using namespace gflags;

DEFINE_int32(port, 8080, "listening port");
DEFINE_int32(dnn_port, 8000, "dnn server port (default 8000)");
DEFINE_string(dnn_hostname, "127.0.0.1", "dnn server hostname (127.0.0.1)");
DEFINE_int32(num_workers, 1, "num workers");

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  ParseCommandLineFlags(&argc, &argv, true);
  
  auto handler = std::make_shared<AppClientHandler>(FLAGS_dnn_hostname, FLAGS_dnn_port);
  auto server = folly::make_unique<ThriftServer>();

  server->setPort(FLAGS_port);
  server->setNWorkerThreads(FLAGS_num_workers);
  server->setInterface(std::move(handler));

  LOG(INFO) << "Listening on: " << FLAGS_port;
  LOG(INFO) << "Sending to: " << FLAGS_dnn_hostname << ":" << FLAGS_dnn_port;
  LOG(INFO) << "Workers: " << FLAGS_num_workers;

  server->serve();

  return 0;
}
