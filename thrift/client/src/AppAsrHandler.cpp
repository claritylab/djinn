
#include "AppClientHandler.h"

#include <folly/wangle/Future.h>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-rbm.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

using namespace folly::wangle;
using namespace apache::thrift;
using namespace apache::thrift::async;
using namespace facebook::windtunnel::treadmill::services::dnn;

using namespace kaldi;
using namespace kaldi::nnet1;

typedef kaldi::int32 int32;

Future<std::unique_ptr<AppResult> > AppClientHandler::future_asr() {

  if (client_ == nullptr) {
    std::shared_ptr<apache::thrift::async::TAsyncSocket> socket(
        apache::thrift::async::TAsyncSocket::newSocket(this->getEventBase(),
                                                             hostname_,
                                                             port_));
    client_ = std::shared_ptr<
                  facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient>(
                    new facebook::windtunnel::treadmill::services::dnn::DnnAsyncClient(
                      std::unique_ptr<apache::thrift::HeaderClientChannel,
                      apache::thrift::TDelayedDestruction::Destructor>(
                          new apache::thrift::HeaderClientChannel(socket)
                        )
                      )
                    );
    // Initialize ASR

    // First load in the raw input data
    kaldi::Matrix<kaldi::BaseFloat> data_in;
    std::string asr_input = "input/asr.data.in";
    kaldi::Input ki(asr_input, false);
    data_in.Read(ki.Stream(), false, false);
    ki.Close();

    // Prepare the dimensions
    dim_t asr_dim;
    asr_dim.n_in = data_in.NumRows();
    asr_dim.c_in = data_in.NumCols();
    asr_dim.w_in = 1; 
    asr_dim.h_in = 1;
    asr_dim.input_len = asr_dim.n_in * asr_dim.c_in;
    dimensions["asr"] = asr_dim;
   
    
    data["asr"] = (double*)malloc(asr_dim.input_len*sizeof(double));

    for(kaldi::MatrixIndexT i = 0; i < data_in.NumRows(); i++){
      for(kaldi::MatrixIndexT j = 0; j < data_in.NumCols(); j++){
        data["asr"][i*data_in.NumCols() + j] = data_in.Row(i)(j); 
      }
    }

    // Second the post-transformation feature data
    std::ifstream feat_file ("input/asr.feat.in");
    char useless;
    feat_file >> useless;
    asr_feat = (double*) malloc(440*548*sizeof(double));
    for(int i = 0; i < 440*548; i++){
      double temp;
      feat_file >> temp;
      asr_feat[i] = temp;
    }
    feat_file.close();

    asr_feat_dim.n_in = 440;
    asr_feat_dim.c_in = 548; 
    asr_feat_dim.w_in = 1;
    asr_feat_dim.h_in = 1;
    asr_feat_dim.input_len = asr_feat_dim.n_in * asr_feat_dim.c_in;


    // Then the feature transformation model
    std::string feature_transf_model = "input/asr.feature_transform";
    nnet_transf.Read(feature_transf_model);

    nnet_transf.SetDropoutRetention(1.0);

  }

  std::cout << "Initialization for ASR done;" << std::endl;
  folly::MoveWrapper<Promise<std::unique_ptr<AppResult> > > promise;
  auto future = promise->getFuture();

  this->getEventBase()->runInEventBaseThread(
    [this, promise]() mutable {
          
    struct timeval app_start_time, app_end_time, app_diff_time;
    struct timeval comm_start_time, comm_end_time, comm_diff_time;

    // start of application
    gettimeofday(&app_start_time, nullptr);

    dim_t dim = dimensions["asr"];
    double* data_in = data["asr"];

    // Prepare transfomration
    CuMatrix<BaseFloat> feats, feats_transf;
    feats.Resize(dim.n_in, dim.c_in);

    // Fill in the input data matrix
    for(MatrixIndexT i = 0; i < feats.NumRows(); i++){
      for(MatrixIndexT j = 0; j < feats.NumCols(); j++){
        feats.Row(i)(j) = data_in[i*feats.NumCols() + j];
      }
    }

    // Do feature transformation
    LOG(INFO) << "Doing feature transformation on input w/ N: " << dim.n_in << " C: " << dim.c_in;
    nnet_transf.Feedforward(feats, &feats_transf);

    // Prepare Work
    Work work;
    work.op = "asr";
    work.n_in = feats_transf.NumRows();
    work.c_in = feats_transf.NumCols();
    work.w_in = 1;
    work.h_in = 1;

    for(MatrixIndexT i = 0; i < feats_transf.NumRows(); i++){
      for(MatrixIndexT j = 0; j < feats_transf.NumCols(); j++){
        work.data.push_back(feats_transf.Row(i)(j));  
      }
    } 

    int network_data_size = work.data.size() * sizeof(float);
    std::cout << "Size is " << work.data.size() << std::endl;

    gettimeofday(&comm_start_time, nullptr);

    auto f = client_->future_fwd(work);

    LOG(INFO) << "Calling DNN service foward pass.";
    f.then(
      [promise, app_start_time, app_end_time, app_diff_time,
      comm_start_time, comm_end_time, comm_diff_time,
      network_data_size](folly::wangle::Try<ServerResult>&& t) mutable {
      // end of application
      Matrix<BaseFloat> nnet_out;
      nnet_out.Resize(548, 1706);

      for(MatrixIndexT i = 0; i < nnet_out.NumRows(); i++){
        for(MatrixIndexT j = 0; j < nnet_out.NumCols(); j++){
          nnet_out.Row(i)(j) = t.value().data[i*nnet_out.NumCols() + j];
        }
      }

      gettimeofday(&app_end_time, nullptr);
      timersub(&app_end_time, &app_start_time, &app_diff_time);

      std::unique_ptr<AppResult> ret(new AppResult());;
      ret->app_time = app_diff_time.tv_sec * 1000 + app_diff_time.tv_usec / 1000;
      
      ret->comm_time = (comm_diff_time.tv_sec * 1000 + comm_diff_time.tv_usec / 1000)
        - t.value().time_ms;
      
      ret->fwd_time = t.value().time_ms;
      ret->comm_data_size = network_data_size;

      promise->setValue(std::move(ret));
      });

      return f;
    });

    return future;
}

