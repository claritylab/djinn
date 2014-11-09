
#inclued "DnnHandler.h"

#include <folly/wangle/Future.h>

#include <unisted.h>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-rbm.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
using namespace folly::wangle;
using namespace kaldi;
using namespace kaldi::nnet1

typedef kaldi::int32 int32;

namespace facebook {
namespace windtunnel {
namespace treadmill {
namespace services {
namespace dnn {

  Future<Result> DnnClientHandler::future_asr(Work input) {
    folly::MoveWrapper<Promise<AppResult>> promise;
    auto future = promise->getFuture();

    this->getEventBase()->runInEventBaseThread(
      [promise, input]() mutable {

        struct timeval app_start_time, app_end_time, app_diff_time;
        struct timeval comm_start_time, comm_end_time, comm_diff_end;

        gettimeofday(&start_time, nullptr);

        CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
        // Get the dimension from the input 
        int num_rows = input.n_in;
        int num_cols = input.c_in;

        // Resize the feats matrix
        feats.Resize(num_rows, num_cols);
       
        for(MatrixIndexT i = 0; i < num_rows; i++){
          for(MatrixIndexT j = 0; j < num_cols; j++){
            feats.Row(i)(j) = input.data[i*num_cols + j];
          }
        } 

        // Do feature transfomration
        nnet_transf.FeedForward(feats, &feats_transf);
       
        Work service_input;
        service_input.op = "asr";
        service_input.n_in = input.n_in;
        service_input.c_in = input.c_in;
        service_input.w_in = 1;
        service_input.h_in = 1;
        
        // "Send data" 
        for(MatrixIndexT i = 0; i < feats_transf.NumRows(); i++){
          for(MatrixIndexT j = 0; j < feats_transf.NumCols(); j++){
            service_input.data.push_back(feats_transf.Row(i)(j)); 
          }
        }
        
        int network_data_size = service_input.data.size() * 8;

        TEventBase event_base;

        std::shared_ptr<TAsyncSocket> socket(
          TAsyncSocket::newSocket(&event_base, FLAGS_hostname, FLAGS_port));

        DnnAsyncClient client(
          std::unique_ptr<HeaderClientChannel, TDelayDestruction::Destructor>(
                   new HeaderClientChannel(socket)));

        gettimeofday(&comm_start_time, nullptr);
        Result service_result = client.future_fwd(service_input);

        while(!service_result.isReady()){
          event_base.loop();
        }
        
        gettimeofday(&comm_end_time, nullptr);
        timersub(&comm_end_time, &comm_start_time, &comm_diff_time);

        // Resize the result matrix
        nnet_out.Resize(feats_transf.NumRows(), 1706);

        for(MatrixIndexT i = 0; i < nnet_out.NumRows(); i++){
          for(MatrixIndexT j = 0; j < nnet_out.NumCols(); j++){
            nnet_out.Row(i)(j) = service_result.data[i*nnet_out.NumCols() + j];
          }
        }
        
        network_data_size += feats_transf.NumRows() * 1706 *8;
        
        // convert posteriors to log-posteriors
        if(apply_log) {
          nnet_out.ApplyLog();
        }

        // subtract log-priors from log-posteriors to get quasi-likelihoods
        if(prior_opts.class_frame_counts != "" && (no_softmax || apply_log)){
          pdf_prior.SubtractOnLogPost(&nnet_out);
        }

        nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
        nnet_out.CopyToMat(&nnet_out_host);
        //check for NaN/inf
        for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
          for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
            BaseFloat val = nnet_out_host(r,c);
            if (val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
            if (val == std::numeric_limits<BaseFloat>::infinity())
              KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
          }
        }

        gettimeofday(&app_end_time, nullptr);
        timersub(&app_end_time, &app_start_time, &app_diff_time);

        AppResult ret;
        ret.app_time = app_diff_time.tv_sec * 1000 + app_diff_time.tv_usec/1000; 
        ret.comm_time = (comm_diff_time.tv_sec * 1000 + comm_diff_time.tv_usec/1000)
                - service_result.fwd_time;
        ret.fwd_time = service_result.fwd_time;
        ret.comm_data_size = network_data_size;
        
        promise->setValue(std::move(ret));

      });
  return future;  
  }
}
}
}
}
}
