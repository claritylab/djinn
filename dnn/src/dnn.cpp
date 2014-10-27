/* Johann Hauswald
 * jahausw@umich.edu
 * 2014
 */

#include <glog/logging.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <stdio.h>
#include <sys/time.h>

#include "dnn.h"
#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"
#include "SENNA_VBS.h"
#include "SENNA_PT0.h"
#include "SENNA_SRL.h"

using namespace std;

Net<float>* SERVICE_init(po::variables_map& vm)
{
    Net<float>* net = new Net<float>(vm["model"].as<string>());
    net->set_debug_info(vm["debug"].as<bool>());

    // some hacky ways to read data. not all models created equal
    // use senna utils to read in model
    // TODO: fix mem leak on senna
    string srv = vm["service"].as<string>();
    string weights = vm["weights"].as<string>();
    if(srv == "imc")
    {
        net->CopyTrainedLayersFrom(weights);
    }
    else if(srv == "asr")
    {
        // Open the file
        ifstream weight_file (weights.c_str());
        if(!weight_file.is_open()){
          cerr << "No weight file found."<<endl;
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
            for(int j = 0; j < 2048; j++){
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
            assert((float_cnt == length) 
                            && "Total Number of biase not equal to expected\n");
            cout<<"Number of bias read in: "<<float_cnt
                    <<" and expected number is " <<length<<endl;
            // Push the bias to caffe
            Blob<float> *ip1_bias = net->layers()[2*i]->blobs()[1].get();
            ip1_bias->set_cpu_data(bias_vec);
            // Then just get rid of the sigmoid line
            getline(weight_file, line);
        }

        // Now the weights and bias are in and set in the neural nets
        // Give control to caffe
        
    }
    else if(srv == "pos")
    {
        SENNA_POS *pos = SENNA_POS_new(weights.c_str());

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(pos->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(pos->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(pos->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(pos->l2_bias);
    }
    else if(srv == "chk")
    {
        SENNA_CHK *chk = SENNA_CHK_new(weights.c_str());

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(chk->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(chk->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(chk->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(chk->l2_bias);
    }
    else if(srv == "ner")
    {
        SENNA_NER *ner = SENNA_NER_new(weights.c_str());

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(ner->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(ner->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(ner->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(ner->l2_bias);
    }
    else if(srv == "vbs")
    {
        SENNA_VBS *vbs = SENNA_VBS_new(weights.c_str());

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(vbs->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(vbs->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(vbs->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(vbs->l2_bias);
    }
    else if(srv == "pt0")
    {
        SENNA_PT0 *pt0 = SENNA_PT0_new(weights.c_str());

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(pt0->l1_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(pt0->l1_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(pt0->l2_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(pt0->l2_bias);
    }
    else if(srv == "srl")
    {
        SENNA_SRL *srl = SENNA_SRL_new(weights.c_str());

        Blob<float> *ip1_weights = net->layers()[0]->blobs()[0].get();
        ip1_weights->set_cpu_data(srl->l3_weight);
        Blob<float> *ip1_bias = net->layers()[0]->blobs()[1].get();
        ip1_bias->set_cpu_data(srl->l3_bias);
        Blob<float> *ip2_weights = net->layers()[2]->blobs()[0].get();
        ip2_weights->set_cpu_data(srl->l4_weight);
        Blob<float> *ip2_bias = net->layers()[2]->blobs()[1].get();
        ip2_bias->set_cpu_data(srl->l4_bias);
    }

    return net;
}

void SERVICE_fwd(float *in, int in_size, float *out, int out_size, Net<float>* net)
{
    float loss;
    LOG(INFO) << in_size;
    vector<Blob<float>* > in_blobs = net->input_blobs();
    in_blobs[0]->set_cpu_data(in);
    vector<Blob<float>* > out_blobs = net->ForwardPrefilled(&loss);
    assert(out_size == out_blobs[0]->count());
    memcpy(out, out_blobs[0]->cpu_data(), out_size*sizeof(float));
}
