#include <pthread.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <glog/logging.h>
#include <ctime>
#include "thread.h"
#include "socket.h"
#include "utils.h"

#define DEBUG 0

using namespace std;

void SERVICE_fwd(float *in, int in_size, float *out, int out_size, Net<float>* net)
{
    float loss;
    vector<Blob<float>* > in_blobs = net->input_blobs();
    in_blobs[0]->set_cpu_data(in);
    vector<Blob<float>* > out_blobs = net->ForwardPrefilled(&loss);
    assert(out_size == out_blobs[0]->count());
    memcpy(out, out_blobs[0]->cpu_data(), out_size*sizeof(float));
}

pthread_t request_thread_init(int sock)
{
  // Prepare to create a new pthread
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 1024*1024);
 // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  
  // Create a new thread starting with the function request_handler
  pthread_t tid;
  if(pthread_create(&tid, &attr, request_handler, (void *)sock) != 0){
    printf("Failed to create a request handler thread.\n");
    return -1;
  }
  return thread_id;
}

void* request_handler(void* sock)
{
  int socknum = (int)sock;
  
  // A client is supposed to follow the following protocol  
  // 1. Send the application type
  // 2. Send the size of input features
  // 3. Loop sending input and receive result

  Net<float>* net;

  // 1. Receive the application type
  int req_type;
  SOCKET_receive(socknum, (char*)&req_type, sizeof(int), DEBUG);  
   
  char* config_file_name = new char[30];
  sprintf(config_file_name, "net-configs/%s.prototxt", request_name[req_type]);
  char* weight_file_name = new char[30];
  sprintf(weight_file_name, "weights/%s.caffemodel", request_name[req_type]);

  // Now we proceed differently based on the type of request
  net = new Net<float>(config_file_name);

  // If you need to update model, uncomment/change the next line and the model name above
  //translate_kaldi_model(weight_file_name, net, true);

  std::clock_t model_ld_start = clock();
  switch(req_type){
    case FACE:
        printf("not implemented\n");
        return 0;
    case ASR: case IMC: case DIG: case POS: case NER: case CHK: case SRL: case VBS: case PT0: {
         net->CopyTrainedLayersFrom(weight_file_name);
         break;
    }
    default:
        printf("Illegal request type\n");
        return -1; 
  }
  std::clock_t model_ld_end = clock(); 

  std::clock_t model_ld_time = model_ld_end - model_ld_start;

  Net<float>* espresso = net;
  // Dump the network
  // caffe::NetParameter output_net_param;
  // espresso->ToProto(&output_net_param, true);
  // WriteProtoToBinaryFile(output_net_param,
  //         weight_file_name + output_net_param.name());
  
  // Now we receive the input data length (in float)
  int sock_elts = SOCKET_rxsize(socknum);
  if(sock_elts < 0){
    printf("Error num incoming elts\n");
    exit(1);
  }

  int n_in = espresso->input_blobs()[0]->num();
  int c_in = espresso->input_blobs()[0]->channels();
  int w_in = espresso->input_blobs()[0]->width();
  int h_in = espresso->input_blobs()[0]->height();
  int in_elts = espresso->input_blobs()[0]->count();
  int n_out = espresso->output_blobs()[0]->num();
  int c_out = espresso->output_blobs()[0]->channels();
  int w_out = espresso->output_blobs()[0]->width();
  int h_out = espresso->output_blobs()[0]->height();
  int out_elts = espresso->output_blobs()[0]->count();
  float *in = (float*) malloc(in_elts * sizeof(float));
  float *out = (float*) malloc(out_elts * sizeof(float));

  // reshape input dims if incoming data > current net config
  // TODO(johann): this is (only) useful for img stuff currently
  std::cout << "Sock elts is " << sock_elts << std::endl;

  if(sock_elts/(c_in*w_in*h_in) > n_in)
  {
      n_in = sock_elts/(c_in*w_in*h_in);
      printf("Reshaping input to dims %d %d %d %d...\n", n_in, c_in, w_in, h_in);
      espresso->input_blobs()[0]->Reshape(n_in, c_in, w_in, h_in);
      in_elts = espresso->input_blobs()[0]->count();
      float *tmp = realloc(in, sock_elts * sizeof(float));
      if(tmp != NULL)
          in = tmp;
      else {
          printf("Can't realloc\n");
          exit(1);
      }

      n_out = n_in;
      printf("Reshaping output to dims %d %d %d %d...\n", n_out, c_out, w_out, h_out);
      espresso->output_blobs()[0]->Reshape(n_out, c_out, w_out, h_out);
      out_elts = espresso->output_blobs()[0]->count();
      tmp = realloc(out, out_elts * sizeof(float));
      if(tmp != NULL)
          out = tmp;
      else {
          printf("Can't realloc\n");
          exit(1);
      }
  }

  // Now we enter the main loop of the thread, following this order
  // 1. Receive input feature (has to be in the size of sock_elts)
  // 2. Do forward pass
  // 3. Send back the result
  // 4. Repeat 1-3
  std::clock_t second_fwd_pass_time = 0;
  std::clock_t first_fwd_pass_time = 0;
  while(1){
    if(DEBUG) printf("Receiving input features from client...\n");
    int rcvd = SOCKET_receive(socknum, (char*) in, in_elts*sizeof(float), DEBUG);
    if(rcvd == 0) break; // Client closed the socket
    
    if(DEBUG) printf("Start neural network forward pass...\n");
    
    std::clock_t fwd_pass_start = std::clock();
    SERVICE_fwd(in, in_elts, out, out_elts, espresso);
    std::clock_t fwd_pass_end = std::clock();
    first_fwd_pass_time = fwd_pass_end - fwd_pass_start;

    fwd_pass_start = std::clock();
    SERVICE_fwd(in, in_elts, out, out_elts, espresso);
    fwd_pass_end = std::clock();
    second_fwd_pass_time = (fwd_pass_end - fwd_pass_start);
    
    if(DEBUG) printf("Sending result back to client...\n");
    SOCKET_send(socknum, (char*) out, out_elts*sizeof(float), DEBUG);
  }

  // Client has finished and close the socket
  // Print timing info
  unsigned int thread_id = (unsigned int) pthread_self();
  printf("Request type: %s, thread ID: %ld\n", request_name[req_type], thread_id);
  printf("Model loading time: %d clock cycles, %.4fms\n", model_ld_time, (1000 * (float)model_ld_time)/CLOCKS_PER_SEC);
  printf("1st Forward pass time: %d clock cycles, %.4fms\n", first_fwd_pass_time, (1000 * (float)first_fwd_pass_time)/CLOCKS_PER_SEC);
  printf("2nd Forward pass time: %d clock cycles, %.4fms\n", second_fwd_pass_time, (1000 * (float)second_fwd_pass_time)/CLOCKS_PER_SEC);
  // Exit the thread

  if(DEBUG) printf("Socket closed by the client. Terminating thread now.\n");

  pthread_mutex_lock(&mutex);
  finished_threads ++;
  pthread_mutex_unlock(&mutex);

  free(in);
  free(out);

  return;
}
