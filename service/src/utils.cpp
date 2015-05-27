// utils.cpp
// Yiping Kang (ypkang@umich.edu)
// 2014
//

#define DEBUG 0

#include "utils.h"

int translate_kaldi_model(char * weight_file_name, Net<float>* net, bool dump)
{
      ifstream weight_file (weight_file_name);
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
            
          if(DEBUG) cout<<"Dimension "<<length<<" "<<width<<endl;
                      
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
          if(DEBUG) cout<<"Number of weights read in: "<<float_cnt
                  <<" and expected number is " <<width*length<<endl;
          assert((float_cnt == (width*length)) 
                          && "Total number of weights not equal to expected\n");
          
          // Push the weight to caffe 
          Blob<float> *ip1_weights = net->layers()[2*i]->blobs()[0].get();
          ip1_weights->set_cpu_data(weight_vec);
            
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
          if(DEBUG) cout<<"Number of bias read in: "<<float_cnt
                <<" and expected number is " <<length<<endl;
          assert((float_cnt == length) 
                          && "Total Number of biase not equal to expected\n");

          // Push the bias to caffe
          Blob<float> *ip1_bias = net->layers()[2*i]->blobs()[1].get();
          ip1_bias->set_cpu_data(bias_vec);
          
          // Last Affine Transform Layer, no sigmoid following
          if(i == 6) break;

          // Then just get rid of the sigmoid line
          getline(weight_file, line);
      }

      if(dump){
        // Dump the network
        caffe::NetParameter output_net_param;
        net->ToProto(&output_net_param, true);
        WriteProtoToBinaryFile(output_net_param,
           weight_file_name + output_net_param.name());
      }

      // Now the weights and bias are in and set in the neural nets
      // Give control to caffe
      return 0;
}
