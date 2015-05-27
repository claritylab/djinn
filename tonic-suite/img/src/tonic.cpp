#include <stdio.h>
#include "tonic.h"

void reshape(Net<float> *net, int input_size) {
  int n_in = net->input_blobs()[0]->num();
  int c_in = net->input_blobs()[0]->channels();
  int w_in = net->input_blobs()[0]->width();
  int h_in = net->input_blobs()[0]->height();

  int n_out = net->output_blobs()[0]->num();
  int c_out = net->output_blobs()[0]->channels();
  int w_out = net->output_blobs()[0]->width();
  int h_out = net->output_blobs()[0]->height();

  int in_elts = net->input_blobs()[0]->count();
  int out_elts = net->output_blobs()[0]->count();

  // assumes C, H, W are known, only reshapes along batch
  if(input_size/(c_in*w_in*h_in) != n_in)
  {
    n_in = input_size/(c_in*w_in*h_in);
    LOG(INFO) << "Reshaping input to dims: "
      << n_in << " " << c_in << " " << w_in << " " << h_in;
    net->input_blobs()[0]->Reshape(n_in, c_in, w_in, h_in);
    in_elts = net->input_blobs()[0]->count();
    // float *tmp = realloc(in, input_size * sizeof(float));
    // if(tmp != NULL)
    //   in = tmp;
    // else {
    //   LOG(ERROR) << "Can't realloc input.";
    //   exit(1);
    // }

    n_out = n_in;
    LOG(INFO) << "Reshaping outnput to dims: "
      << n_out << " " << c_out << " " << w_out << " " << h_out;
    net->output_blobs()[0]->Reshape(n_out, c_out, w_out, h_out);
    out_elts = net->output_blobs()[0]->count();
    // tmp = realloc(out, out_elts * sizeof(float));
    // if(tmp != NULL)
    //   out = tmp;
    // else {
    //   LOG(ERROR) << "Can't realloc output.";
    //   exit(1);
    // }
  }
}
