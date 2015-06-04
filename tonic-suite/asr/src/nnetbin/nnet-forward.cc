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

/* Yiping Kang
 * ypkang@umich.edu
 * 2014
 */

#include <limits>
#include <ctime>
#include <fstream>
#include <vector>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-rbm.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "socket.h"
#include "tonic.h"

#define DEBUG 0

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {

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
  
    // Local inference information
    string common = "../../common/";
    po.Register("common", &common, "Directory with configs and weights");
    string network = "asr.prototxt";
    po.Register("network", &network, "Network config file (.prototxt)");
    string weights = "asr.caffemodel";
    po.Register("weights", &weights, "Pretrained weights (.caffemodel)");

    // DjiNN service information
    bool djinn = false;
    po.Register("djinn", &djinn, "Use DjiNN service?");
    string hostname = "localhost";
    po.Register("hostname", &hostname, "Server IP addr");
    int portno = 8080;
    po.Register("portno", &portno, "Server port");

    // Common configuraition
    bool gpu = "false";
    po.Register("gpu", &gpu, "Use GPU?");
    bool debug = "false";
    po.Register("debug", &debug, "Turn on all debug");

    // ASR specific inputs and flags
    // inherited from Kaldi
    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");
    bool no_softmax = false;
//    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
//    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    // Read in the argument
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    // Input to ASR
    // inherited from Kaldi
    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);

    // Initialize tonic app
    TonicSuiteApp app;
    app.task = "asr";
    app.network = network;
    app.weights = weights;

    app.djinn = djinn;
    app.gpu = gpu;

    if(app.djinn) {
      app.hostname = hostname;
      app.portno = portno;
      app.socketfd = CLIENT_init((char*)app.hostname.c_str(), app.portno, debug);

      if (app.socketfd < 0){
        exit(1);
      }
    }else{
      app.net = new Net<float>(app.network);
      app.net->CopyTrainedLayersFrom(app.weights);
      Caffe::set_phase(Caffe::TEST);
      if(app.gpu)
        Caffe::set_mode(Caffe::GPU);
      else
        Caffe::set_mode(Caffe::CPU);
    }

    // Set request type
    app.pl.size = 0;
    strcpy(app.pl.req_name, app.task.c_str());


//    //Select the GPU
//#if HAVE_CUDA==1
//    CuDevice::Instantiate().SelectGpuId(use_gpu);
//    CuDevice::Instantiate().DisableCaching();
//#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    //optionally remove softmax
//    if (no_softmax && nnet.GetComponent(nnet.NumComponents()-1).GetType() == Component::kSoftmax) {
//      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
//      nnet.RemoveComponent(nnet.NumComponents()-1);
//    }
//    //check for some non-sense option combinations
//    if (apply_log && no_softmax) {
//      KALDI_ERR << "Nonsense option combination : --apply-log=true and --no-softmax=true";
//    }
//    if (apply_log && nnet.GetComponent(nnet.NumComponents()-1).GetType() != Component::kSoftmax) {
//      KALDI_ERR << "Used --apply-log=true, but nnet " << model_filename 
//                << " does not have <softmax> as last component!";
//    }
//    
    PdfPrior pdf_prior(prior_opts);
    if (prior_opts.class_frame_counts != "" && (!no_softmax && !apply_log)) {
      KALDI_ERR << "Option --class-frame-counts has to be used together with "
                << "--no-softmax or --apply-log";
    }

    // disable dropout
//    nnet_transf.SetDropoutRetention(1.0);
//    nnet.SetDropoutRetention(1.0);

  
//    if(write_model_file != ""){
//      // So we write the model to file
//      // and exit the program
//      Output ko(write_model_file, false); 
//      nnet.Write(ko.Stream(), false);
//      exit(0);
//    } 
//
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;
    Matrix<BaseFloat> recv_mat;

    Timer time;
//    double time_now = 0;
//    double time_comm = 0;
//    double time_nn = 0;
//    double time_read_feat = 0;
    int32 num_done = 0;
    
    // iterate over all feature files
    // cumulate them for batch processing
    vector<float> batched_feats;
    int offset = 0;
    int total_rows = 0;
    std::vector<int> feats_row_cnt;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      const Matrix<BaseFloat> &mat = feature_reader.Value();

      KALDI_LOG << "Processing utterance " << num_done+1 
                    << ", " << feature_reader.Key() 
                    << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      BaseFloat sum = mat.Sum();
      if (!KALDI_ISFINITE(sum)) {
        KALDI_ERR << "NaN or inf found in features of " << feature_reader.Key();
      }
      
      // Feature transformation will use GPU if possible
      feats = mat;
      
      // Preprocessing feature transformation      
      nnet_transf.Feedforward(feats, &feats_transf);

      int cur_num_feats = feats_transf.NumCols() * feats_transf.NumRows();
      batched_feats.resize(batched_feats.size() + cur_num_feats);

      feats_row_cnt.push_back(feats_transf.NumRows());
      
      // Concatenate this to the total input
      for(MatrixIndexT i = 0; i < feats_transf.NumRows(); i++){
        std::copy(feats_transf.Row(i).Data(), feats_transf.Row(i).Data()+feats_transf.NumCols(), batched_feats.begin()+offset);
        offset += feats_transf.NumCols();
        total_rows++;
      }
    }
    
    app.pl.num = total_rows;
    app.pl.size = offset / total_rows;
    app.pl.data = (float*) malloc(offset * sizeof(float));
    memcpy((char*)app.pl.data, (char*)&batched_feats[0], offset*sizeof(float));

    std::vector<Matrix<BaseFloat> > output_list;
    // Inference
    if(app.djinn){
      KALDI_LOG << "Use DjiNN service to inference.";

      // Send request type
      SOCKET_send(app.socketfd, (char*)&app.pl.req_name, MAX_REQ_SIZE, debug);
      
      // Resize the receiving matrix
      recv_mat.Resize(total_rows, 1706);
      nnet_out.Resize(feats_transf.NumRows(), 1706); 

      // Send data length
      SOCKET_txsize(app.socketfd, offset);

      // Send features
      SOCKET_send(app.socketfd, (char*)app.pl.data, offset*sizeof(float), debug);

      // Receive results 
      // Receive into multiple kaldi's matrix format 
      int total_rcvd = 0;
      for(int feat_idx=0; feat_idx < feats_row_cnt.size(); feat_idx++){
        Matrix<BaseFloat> nnet_out;
        nnet_out.Resize(feats_row_cnt[feat_idx], 1706);
        for(MatrixIndexT i = 0; i < nnet_out.NumRows(); i++){
          int rcvd = SOCKET_receive(app.socketfd, (char*)nnet_out.Row(i).Data(), 1706 * sizeof(float), DEBUG);
          total_rcvd += rcvd; 
        }
        output_list.push_back(nnet_out);
      }
      assert(total_rcvd == total_rows * 1706 * sizeof(float) && "Not recving enough features");
      
      // Close the socket
      SOCKET_close(app.socketfd,DEBUG);
      KALDI_LOG << "DjiNN service returns. Socket closed.";
    }else{
      // Use local inference
      KALDI_LOG << "Use local dnn service to inference.";
//      nnet.Feedforward(feats_transf, &nnet_out);
      float loss;
      reshape(app.net, offset);

      float* preds = (float*)malloc(offset*sizeof(float));
      vector<Blob<float>* > in_blobs = app.net->input_blobs();
      in_blobs[0]->set_cpu_data((float*)app.pl.data);

      vector<Blob<float>* > out_blobs = app.net->ForwardPrefilled(&loss);

      memcpy(preds, out_blobs[0]->cpu_data(), offset*sizeof(float));

      // Copy into multiple kaldi's matrix format
      int cpy_offset = 0;
      for(int feat_idx = 0; feat_idx < feats_row_cnt.size(); feat_idx++){
        Matrix<BaseFloat> nnet_out;
        nnet_out.Resize(feats_row_cnt[feat_idx], 1706);
        Vector<BaseFloat> data_vec;
        memcpy((char*)(data_vec.Data()+cpy_offset), (char*)(preds+cpy_offset), feats_row_cnt[feat_idx]*1706*sizeof(float) );
        nnet_out.CopyRowsFromVec(data_vec);
        output_list.push_back(nnet_out);
      }
    }
   // 
   // // convert posteriors to log-posteriors
   // if (apply_log) {
   //   nnet_out.ApplyLog();
   // }
    
    // subtract log-priors from log-posteriors to get quasi-likelihoods
    if (prior_opts.class_frame_counts != "" && (no_softmax || apply_log)) {
      pdf_prior.SubtractOnLogpost(&nnet_out);
    }
    
    //check for NaN/inf
//    for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
//      for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
//        BaseFloat val = nnet_out_host(r,c);
//        if (val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
//        if (val == std::numeric_limits<BaseFloat>::infinity())
//          KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
//      }
//    }
      
    // Iterate over feature reader again and write the output  

    SequentialBaseFloatMatrixReader output_feature_reader(feature_rspecifier);
    int cnt = 0;
    for (; !output_feature_reader.Done(); output_feature_reader.Next()) {
      feature_writer.Write(output_feature_reader.Key(), output_list[cnt]);
      cnt++;
    }

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
