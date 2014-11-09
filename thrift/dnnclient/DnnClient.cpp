#include <memory>
#include <gflags/gflags.h>

#include "thrift/lib/cpp2/server/ThriftServer.h"

#include "DnnClientHandler.h"
#include "caffe/caffe.hpp"

using namespace apache::thrift;
using namespace apache::thrift::async;

using namespace boost;

using namespace dnn;

DEFINE_bool(debug, false, "debug");
DEFINE_int32(port, 8080, "server port (default: 8080)");
DEFINE_string(gpu_mode, "pool", 
                "mode of gpu configuration (default: pool - pick which ever one is free.)")

int main(int argc, char* argv[]) {
        
  google::InitGoogleLogging(argv[0]);

  google::ParseCommandLineFlags(&argc, &argv, true);
  
  auto handler = std::make_shared<DnnClientHandler>();
  auto server = folly::make_unique<ThriftServer>();

  server->setPort(FLAGS_port);
  server->setNWorkerThreads(FLAGS_num_workers);
  server->setInterface(std::move(handler));

  server->serve();

  return 0;
}
