// nnetbin/nnet-forward.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <limits>
#include <ctime>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-rbm.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

#include "socket.h"
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "Dnn.h"
#include "dnn_types.h"
#include "dnn_constants.h"

#define DEBUG 0

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using namespace dnn;
using namespace boost;


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {

    // YK
    // Socket stuff to talk to the server
   // 
    const char *usage =
        "Perform forward pass through Neural Network.\n"
        "\n"
        "Usage:  nnet-forward [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        " [options] --hostname --portno"
        "e.g.: \n"
        " nnet-forward nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    bool use_service = false;
    po.Register("use-service", &use_service, "Will use the remote dnn service");

    std::string write_model_file("");
    po.Register("write-model", &write_model_file, "nnet-forward is wrtting weights and bias to file");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 


    std::string hostname("");
    po.Register("hostname", &hostname, "Server address of the NN service");
     
    int portno = -1;
    po.Register("portno", &portno, "Port number of the NN service");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;
    
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    //optionally remove softmax
    if (no_softmax && nnet.GetComponent(nnet.NumComponents()-1).GetType() == Component::kSoftmax) {
      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
      nnet.RemoveComponent(nnet.NumComponents()-1);
    }
    //check for some non-sense option combinations
    if (apply_log && no_softmax) {
      KALDI_ERR << "Nonsense option combination : --apply-log=true and --no-softmax=true";
    }
    if (apply_log && nnet.GetComponent(nnet.NumComponents()-1).GetType() != Component::kSoftmax) {
      KALDI_ERR << "Used --apply-log=true, but nnet " << model_filename 
                << " does not have <softmax> as last component!";
    }
    
    PdfPrior pdf_prior(prior_opts);
    if (prior_opts.class_frame_counts != "" && (!no_softmax && !apply_log)) {
      KALDI_ERR << "Option --class-frame-counts has to be used together with "
                << "--no-softmax or --apply-log";
    }

    // disable dropout
    nnet_transf.SetDropoutRetention(1.0);
    nnet.SetDropoutRetention(1.0);

  
    if(write_model_file != ""){
      // So we write the model to file
      // and exit the program
      Output ko(write_model_file, false); 
      nnet.Write(ko.Stream(), false);
      exit(0);
    } 

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;
    Matrix<BaseFloat> recv_mat;

    Timer time;
    double time_now = 0;
    double time_comm = 0;
    double time_nn = 0;
    int32 num_done = 0;
    
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      KALDI_LOG << "Processing utterance " << num_done+1 
                    << ", " << feature_reader.Key() 
                    << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      BaseFloat sum = mat.Sum();
      if (!KALDI_ISFINITE(sum)) {
        KALDI_ERR << "NaN or inf found in features of " << feature_reader.Key();
      }
      
      // push it to gpu
      feats = mat;
      // fwd-pass
      // Preprocessing feature transformation      
      nnet_transf.Feedforward(feats, &feats_transf);
      KALDI_LOG << "Input feature with dimension: "<<feats_transf.NumCols(); 
      if(use_service){
        shared_ptr<TTransport> socket(new TSocket(hostname, portno));
        shared_ptr<TTransport> transport(new TBufferedTransport(socket));
        shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));

        DnnClient client(protocol);

        transport->open();
        
        Work work;
        work.op = "asr";
        work.c_in = 440;
        work.n_in = feats_transf.NumRows();
        work.w_in = 1;
        work.h_in = 1;

        // "Send data"
        for(MatrixIndexT i = 0; i < feats_transf.NumRows(); i++){
          for(MatrixIndexT j = 0; j < feats_transf.NumCols(); j++){
            work.data.push_back(feats_transf.Row(i)(j)); 
          }
        }

        // "Forward pass"
        vector<double> inc;
        client.fwd(inc, work);

        // Resize the result matrix
        nnet_out.Resize(feats_transf.NumRows(), 1706); 

        for(MatrixIndexT i = 0; i < nnet_out.NumRows(); i++){
          for(MatrixIndexT j = 0; j < nnet_out.NumCols(); j++){
            nnet_out.Row(i)(j) = inc[i*nnet_out.NumCols() + j];
          }
        }

        transport->close();

/*        KALDI_LOG << "Use remote dnn service to inference.";
        // Connect to the server
        int socket;
        char* hostname_cstr = new char [hostname.length() + 1];
        std::strcpy(hostname_cstr, hostname.c_str());
        socket = CLIENT_init(hostname_cstr, portno, DEBUG);
        if(socket < 0){
          KALDI_ERR << "Socket return as zero.";
          exit(1);
        }  
        KALDI_LOG << "Establish socket with server at "
                << hostname << ":" << portno;
        // 1. Send the request type (1)
        int req_type = 3;
        SOCKET_send(socket, (char*)&req_type, sizeof(int), DEBUG);
        KALDI_LOG << "Send request type: "<<req_type; 

        // 5. Send the length of the input feature 
        recv_mat.Resize(feats_transf.NumRows(), 1706);

        int total_input_features = feats_transf.NumCols() * feats_transf.NumRows();
        SOCKET_txsize(socket, total_input_features);

        // Now we send the entire sentence over for batch processing
        int total_sent = 0;
        nnet_out.Resize(feats_transf.NumRows(), 1706); 
        
        Timer time_comm_temp;

        for(MatrixIndexT i = 0; i < feats_transf.NumRows(); i++){
          int sent = SOCKET_send(socket, (char*)feats_transf.Row(i).Data(),
                        feats_transf.NumCols() * sizeof(float), DEBUG);
          total_sent += sent;
        }
        assert(total_sent == total_input_features*sizeof(float) && "Not sending enough features.");

        // Now we receive the result, again with the matrix as a whole
        int total_rcvd = 0;
        for(MatrixIndexT i = 0; i < feats_transf.NumRows(); i++){
         int rcvd = SOCKET_receive(socket, (char*)nnet_out.Row(i).Data(),
                        1706 * sizeof(float), DEBUG);
          total_rcvd += rcvd; 
        }
        assert(total_rcvd == feats_transf.NumRows() * 1706 * sizeof(float) && "Not recving enough features");
        
        time_comm += time_comm_temp.Elapsed();

        // Close the socket, don't need it anymore
        SOCKET_close(socket,DEBUG);
        KALDI_LOG << "DNN service finishes. Socket closed.";
        */
      }else{
        // Use local(kaldi's) dnn to inference
        KALDI_LOG << "Use local dnn service to inference.";
        Timer nn_timer;
        nnet.Feedforward(feats_transf, &nnet_out);
        time_nn += nn_timer.Elapsed();
      }
      

      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }
     
      // subtract log-priors from log-posteriors to get quasi-likelihoods
      if (prior_opts.class_frame_counts != "" && (no_softmax || apply_log)) {
        pdf_prior.SubtractOnLogpost(&nnet_out);
      }
     
      //download from GPU
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
      
     // write
      feature_writer.Write(feature_reader.Key(), nnet_out_host);
      
      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 
    KALDI_LOG << "Total app time is " << time_now*1000 << "ms";
    KALDI_LOG << "Communication time is " << time_comm*1000 << "ms";
    KALDI_LOG << "Local neural net time is " << time_nn*1000 << "ms";

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
