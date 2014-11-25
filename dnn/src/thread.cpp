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
#include <cmath>

#include <boost/chrono/thread_clock.hpp>

using namespace std;

extern std::vector<std::string> reqs;
extern map<string, Net<float>* > nets;
// extern float* in;
// extern float* out;
extern int NUM_QS;

#define DEBUG 0

double SERVICE_fwd(float *in, int in_size, float *out, int out_size, Net<float>* net)
{
    float loss;
    struct timeval start, end, diff;
    vector<Blob<float>* > in_blobs = net->input_blobs();

    vector<Blob<float>* > out_blobs;

    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_QS; ++i) {
        in[0] += 1;
        in_blobs[0]->set_cpu_data(in);
        out_blobs = net->ForwardPrefilled(&loss);
        memcpy(out, out_blobs[0]->cpu_data(), sizeof(float));
    }

    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    if(out_size != out_blobs[0]->count())
        LOG(FATAL) << "out_size =! out_blobs[0]->count())";
    else
        memcpy(out, out_blobs[0]->cpu_data(), out_size*sizeof(float));

    return (((double)diff.tv_sec*(double)1000 + (double)diff.tv_usec/(double)1000)/(double)NUM_QS);
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
  return tid;
}

void* request_handler(void* sock)
{
    int socknum = (int)sock;

    // A client is supposed to follow the following protocol  
    // 1. Send the application type
    // 2. Send the size of input featuees
    // 3. Loop sending input and receive result

    int req_type;
    SOCKET_receive(socknum, (char*)&req_type, sizeof(int), DEBUG);  
    LOG(INFO) << "Task " << reqs[req_type] << " forward pass. Total queries " << NUM_QS;

    char* config_file_name = new char[30];
    sprintf(config_file_name, "net-configs/%s.prototxt", request_name[req_type]);
    char* weight_file_name = new char[30];
    sprintf(weight_file_name, "weights/%s.caffemodel", request_name[req_type]);

    // If you need to update model, uncomment/change the next line and the model name above
    //translate_kaldi_model(weight_file_name, net, true);

    std::clock_t model_ld_start = clock();
    // espresso->ShareTrainedLayersWith(nets[reqs[req_type]]);
    std::clock_t model_ld_end = clock();
    std::clock_t model_ld_time = model_ld_end - model_ld_start;

    // Dump the network
    // caffe::NetParameter output_net_param;
    // espresso->ToProto(&output_net_param, true);
    // WriteProtoToBinaryFile(output_net_param,
    //         weight_file_name + output_net_param.name());

    // Now we receive the input data length (in float)
    int sock_elts = SOCKET_rxsize(socknum);
    if(sock_elts < 0){
        LOG(ERROR) << "Error num incoming elts.";
        exit(1);
    }

    int n_in = nets[reqs[req_type]]->input_blobs()[0]->num();
    int c_in = nets[reqs[req_type]]->input_blobs()[0]->channels();
    int w_in = nets[reqs[req_type]]->input_blobs()[0]->width();
    int h_in = nets[reqs[req_type]]->input_blobs()[0]->height();
    int n_out = nets[reqs[req_type]]->output_blobs()[0]->num();
    int c_out = nets[reqs[req_type]]->output_blobs()[0]->channels();
    int w_out = nets[reqs[req_type]]->output_blobs()[0]->width();
    int h_out = nets[reqs[req_type]]->output_blobs()[0]->height();

    int in_elts = nets[reqs[req_type]]->input_blobs()[0]->count();
    int out_elts = nets[reqs[req_type]]->output_blobs()[0]->count();
    float *in = (float*) malloc(in_elts * sizeof(float));
    float *out = (float*) malloc(out_elts * sizeof(float));

    // reshape input dims if incoming data > current net config
    // TODO(johann): this is (only) useful for img stuff currently
    LOG(INFO) << "Elements received on socket " << sock_elts << std::endl;

    if(sock_elts/(c_in*w_in*h_in) != n_in)
    {
        n_in = sock_elts/(c_in*w_in*h_in);
        LOG(INFO) << "Reshaping input to dims: "
            << n_in << " " << c_in << " " << w_in << " " << h_in;
        nets[reqs[req_type]]->input_blobs()[0]->Reshape(n_in, c_in, w_in, h_in);
        in_elts = nets[reqs[req_type]]->input_blobs()[0]->count();
        float *tmp = realloc(in, sock_elts * sizeof(float));
        if(tmp != NULL)
            in = tmp;
        else {
            LOG(ERROR) << "Can't realloc input.";
            exit(1);
        }

        n_out = n_in;
        LOG(INFO) << "Reshaping input to dims: "
                  << n_out << " " << c_out << " " << w_out << " " << h_out;
        nets[reqs[req_type]]->output_blobs()[0]->Reshape(n_out, c_out, w_out, h_out);
        out_elts = nets[reqs[req_type]]->output_blobs()[0]->count();
        tmp = realloc(out, out_elts * sizeof(float));
        if(tmp != NULL)
            out = tmp;
        else {
            LOG(ERROR) << "Can't realloc output.";
            exit(1);
        }
    }

    // Now we enter the main loop of the thread, following this order
    // 1. Receive input feature (has to be in the size of sock_elts)
    // 2. Do forward pass
    // 3. Send back the result
    // 4. Repeat 1-3
    double fwd_pass_time = 0;
    struct timeval start, end, diff;

    bool warmup = true;
    int counter = 0;

    while(1) {
        if(DEBUG) printf("Receiving input features from client...\n");
        LOG(INFO) << "Reading from socket.";
        int rcvd = SOCKET_receive(socknum, (char*) in, in_elts*sizeof(float), DEBUG);
        LOG(INFO) << "Done reading from socket.";

        if(rcvd == 0) break; // Client closed the socket

        if(DEBUG) printf("Start neural network forward pass...\n");
        if(warmup) {
            float loss;
            vector<Blob<float>* > in_blobs = nets[reqs[req_type]]->input_blobs();
            in_blobs[0]->set_cpu_data(in);
            vector<Blob<float>* > out_blobs;
            out_blobs = nets[reqs[req_type]]->ForwardPrefilled(&loss);
            warmup = false;
        }

        LOG(INFO) << "FWD pass start";
        fwd_pass_time += SERVICE_fwd(in, in_elts, out, out_elts, nets[reqs[req_type]]);
        LOG(INFO) << "FWD pass done";

        if(DEBUG) printf("Sending result back to client...\n");
        LOG(INFO) << "Writing to socket.";
        SOCKET_send(socknum, (char*) out, out_elts*sizeof(float), DEBUG);
        LOG(INFO) << "Done writing to socket.";
        counter++;
    }

    // Client has finished and close the socket
    // Print timing info to csv

    unsigned int thread_id = (unsigned int) pthread_self();

    pthread_mutex_lock(&csv_lock);
    FILE* csv_file = fopen(csv_file_name.c_str(), "a");
    int numquery = in_elts;
    if(request_name[req_type] == "pos" ||
            request_name[req_type] == "chk" ||
            request_name[req_type] == "vbs" ||
            request_name[req_type] == "pt0")
        numquery = in_elts/8400; // 28 *300
    else if(request_name[req_type] == "ner")
        numquery = in_elts/10500;
    else if(request_name[req_type] == "srl")
        numquery = (int)sqrt(counter/112);
    else if(request_name[req_type] == "asr")
        numquery = in_elts/(548*440);
    else if(request_name[req_type] == "dig")
        numquery = in_elts/(100*28*28);
    else if(request_name[req_type] == "imc")
        numquery = in_elts/(3*227*227);
    else if(request_name[req_type] == "face")
        numquery = in_elts/(3*152*152);

    fprintf(csv_file, "%s,%s,%d,%.4f,%.4f\n", request_name[req_type],
                                              platform.c_str(),
                                              numquery,
                                              fwd_pass_time/(double)counter,
                                              (double)numquery/(double)fwd_pass_time*(double)counter);

    fclose(csv_file);
    pthread_mutex_unlock(&csv_lock);

    // Exit the thread
    if(DEBUG) printf("Socket closed by the client. Terminating thread now.\n");

    free(in);
    free(out);
    // delete espresso;

    return;
}
