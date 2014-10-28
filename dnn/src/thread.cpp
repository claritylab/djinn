#include "thread.h"
#include <pthread.h>
#include "socket.h"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <glog/logging.h>

#define DEBUG 0

using namespace std;


int request_thread_init(int sock)
{
  // Prepare to create a new pthread
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 1024*1024);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  
  // Create a new thread starting with the function request_handler
  pthread_t thread_id;
  if(pthread_create(&thread_id, &attr, request_handler, (void *)sock) != 0){
    printf("Failed to create a request handler thread.\n");
    return -1;
  }
  return 0;
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
  sprintf(weight_file_name, "weights/%s.dat", request_name[req_type]);

  // Now we proceed differently based on the type of request
  net = new Net<float>(config_file_name);

  switch(req_type){
    
    case IMC:{
      net->CopyTrainedLayersFrom(weight_file_name);
      break;
    }
    case ASR:{
      ifstream weight_file (weight_file_name);
      printf("%s\n", weight_file_name);
      if(!weight_file.is_open()){
        printf("No weight file found.\n");
        exit(1);
      }
      // Read in line by line
      std::string line;
      getline(weight_file, line); // <Nnet>
      for(int i = 0; i < 7; i++){ // 6 is the number of hidden layers
          getline(weight_file, line); // <AffineTransform>
          // Split the first line to get the dimension of the weights matrix
          int loc = line.find_first_of(" ");
          line = line.substr(loc+1, line.length() - (loc+1));
          
          loc = line.find_first_of(" ");
          int length = atoi(line.substr(0, loc).c_str());

          line = line.substr(loc+1, line.length() - (loc+1));
          int width = atoi(line.c_str());
            
          cout<<"Dimension "<<length<<" "<<width<<endl;
                      
          getline(weight_file, line); // <AffineTransform::param>
  
          // Initialize the large weights array
          float* weight_vec = (float*)malloc(sizeof(float)*width*length);
          int float_cnt = 0;
          for(int j = 0; j < length; j++){
            getline(weight_file, line);
            // String stream out each float in the line
            std::stringstream sstream;
            sstream.str(line);
            float num;
            while(sstream >> num){
              weight_vec[float_cnt] = num; 
              float_cnt++;
            }
          }
          // Check if the total number of weights read-in is correct
          cout<<"Number of weights read in: "<<float_cnt
                  <<" and expected number is " <<width*length<<endl;
          assert((float_cnt == (width*length)) 
                          && "Total number of weights not equal to expected\n");
          // Push the weight to caffe 
          Blob<float> *ip1_weights = net->layers()[2*i]->blobs()[0].get();
          ip1_weights->set_cpu_data(weight_vec);
            

          if(i == 6) break;
          // Initialize the less large bias array
          float* bias_vec = (float*) malloc(sizeof(float)*length);
          float_cnt = 0;
          getline(weight_file, line);
          std::stringstream sstream;
          sstream.str(line);
          float num;
          // Get rid of the [ at the begining of the line
          char trash;
          sstream >> trash;

          // Now starts read float
          while(sstream >> num){
            bias_vec[float_cnt] = num;
            float_cnt++;
          } 

          // Check if the total number of bias read-in is correct
          cout<<"Number of bias read in: "<<float_cnt
                <<" and expected number is " <<length<<endl;
          assert((float_cnt == length) 
                          && "Total Number of biase not equal to expected\n");
          // Push the bias to caffe
          Blob<float> *ip1_bias = net->layers()[2*i]->blobs()[1].get();
          ip1_bias->set_cpu_data(bias_vec);
          
          // Then just get rid of the sigmoid line
          getline(weight_file, line);
      }

      // Now the weights and bias are in and set in the neural nets
      // Give control to caffe
      break; 
    }
    case POS:{

        SENNA_POS *pos = SENNA_POS_new(weight_file_name);

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(pos->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(pos->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(pos->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(pos->l2_bias);
        
        break;  
    }
    case NER:{

        SENNA_NER *ner = SENNA_NER_new(weight_file_name);

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(ner->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(ner->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(ner->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(ner->l2_bias);
        
        break; 
    }
    case CHK:{

        SENNA_CHK *chk = SENNA_CHK_new(weight_file_name);

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(chk->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(chk->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(chk->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(chk->l2_bias);
           
        break; 
    }
    case SRL:{

        SENNA_SRL *srl = SENNA_SRL_new(weight_file_name);

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(srl->l3_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(srl->l3_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(srl->l4_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(srl->l4_bias);
        break;
    }
    case VBS:{
  
        SENNA_VBS *vbs = SENNA_VBS_new(weight_file_name);

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(vbs->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(vbs->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(vbs->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(vbs->l2_bias);
        break; 
    }
    case PT0:{
        
        SENNA_PT0 *pt0 = SENNA_PT0_new(weight_file_name);

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(pt0->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(pt0->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(pt0->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(pt0->l2_bias);
        break;
    }
    case FACE:
    case DIG:
    default:
           printf("Illegal request type\n");
           return -1; 
  }

  Net<float>* espresso = net;

  // get elts for top layer
  int in_elts = espresso->input_blobs()[0]->count();
  int out_elts = espresso->output_blobs()[0]->count();
  float *in = (float*) malloc(in_elts * sizeof(float));
  float *out = (float*) malloc(out_elts * sizeof(float));

  // Now we receive the input data length (in float)
  int sock_elts = SOCKET_rxsize(socknum);
  cout << sock_elts << endl;
  if(sock_elts < 0){
    printf("Error num incoming elts\n");
    exit(1);
  }

  // Now we enter the main loop of the thread, following this order
  // 1. Receive input feature (has to be in the size of sock_elts)
  // 2. Do forward pass
  // 3. Send back the result
  // 4. Repeat 1-3
  while(1){
    int rcvd = SOCKET_receive(socknum, (char*) in, in_elts*sizeof(float), DEBUG);
    if(rcvd == 0) break; // Client closed the socket
    SERVICE_fwd(in, in_elts, out, out_elts, espresso);
    SOCKET_send(socknum, (char*) out, out_elts*sizeof(float), DEBUG);
  }

  // Client has finished and close the socket
  // Exit the thread

  if(DEBUG) printf("Socket closed by the client. Terminating thread now.\n");

  free(in);
  free(out);

  return;
}
