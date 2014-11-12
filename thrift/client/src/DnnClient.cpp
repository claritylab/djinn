#include <memory>
#include <gflags/gflags.h>

#include "thrift/lib/cpp2/server/ThriftServer.h"

#include "AppClientHandler.h"
#include "caffe/caffe.hpp"

using namespace apache::thrift;
using namespace apache::thrift::async;
using namespace facebook::windtunnel::treadmill::services::dnn;

using namespace boost;
using namespace gflags;

DEFINE_int32(port, 8080, "server port (default: 8080)");
DEFINE_int32(num_workers, 1, "num workers");

int main(int argc, char* argv[])
{
  ParseCommandLineFlags(&argc, &argv, true);
  
  auto handler = std::make_shared<AppClientHandler>();
  auto server = folly::make_unique<ThriftServer>();

  server->setPort(FLAGS_port);
  server->setNWorkerThreads(FLAGS_num_workers);
  server->setInterface(std::move(handler));

  server->serve();

  return 0;
}
