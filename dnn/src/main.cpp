/* Johann Hauswald
 * jahausw@umich.edu
 * 2014
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>   
#include <unistd.h>  
#include <errno.h>
#include <map>
#include <glog/logging.h>
#include <sys/stat.h>

#include "boost/program_options.hpp" 
#include "socket.h"
#include "thread.h"

using namespace std;
namespace po = boost::program_options;

std::string csv_file_name;
pthread_mutex_t csv_lock;
std::string platform;

std::vector<std::string> reqs;
map<string, Net<float>* > nets;
// float *in;
// float *out;
int NUM_QS;
int openblas_threads;
float cpufreq;

po::variables_map parse_opts( int ac, char** av )
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("server,s", po::value<bool>()->default_value(false), "If specified true, open brainiac as a server")
        ("gpu,u", po::value<bool>()->default_value(true), "Use GPU?")
        ("debug,v", po::value<bool>()->default_value(false), "Turn on all debug")
        ("csv,c", po::value<string>()->default_value("./timing.csv"), "CSV file to put the timings in.")
        ("trial,r", po::value<int>()->default_value(1), "Number of runs to average across (default: 1)")
        ("cpufreq,f", po::value<int>()->default_value(2320500), "The cpu frequency (default: 2.3GHz)")
        ("cputhread,h", po::value<int>()->default_value(0), "CPU threads used (default: 0)")
        ("verbose,b", po::value<bool>()->default_value(false), "Print more info to csv")
        ("transfer, e", po::value<bool>()->default_value(false), "Include data transfer time between host and device in timing (default: false)")
        
        // Options for server setup
        ("portno,p", po::value<int>()->default_value(8080), "Open server on port (default: 8080)")
        ("threadcnt,t", po::value<int>()->default_value(0), "Number of threads to spawn before exiting the server.")

        // Options for local setup
        ("network,n", po::value<string>()->default_value("undefined"), "DNN network to use in this experiment")
        ("input,i", po::value<string>()->default_value("undefined"), "Input to the DNN")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        cout << desc << "\n";
        return vm;
    }
    return vm;
}

// Check if a file already exists 
bool ifexist(std::string filename)
{
  struct stat buf;
  
  if(stat(filename.c_str(), &buf) != -1)
    return true;
  else
    return false;
}

int main(int argc , char *argv[])
{
    // google::InitGoogleLogging(argv[0]);

    po::variables_map vm = parse_opts(argc, argv);
    caffe::Phase phase = 1; // Set phase to TEST

    // Added for 570 project experiments
    // These two numbers (thread, cpufreq) do not have real effect
    // just for csv output purpose 
    openblas_threads = 0; // default to zero
    cpufreq = (float) vm["cpufreq"].as<int>() / (float) 1000000;

    if(vm["gpu"].as<bool>()){
        Caffe::set_mode(Caffe::GPU);
        platform = "gpu";
    }
    else{
        Caffe::set_mode(Caffe::CPU);
        platform = "cpu";
        openblas_threads = vm["cputhread"].as<int>();
    }

    if(vm["server"].as<bool>()){
      std::ifstream file ("nets.txt");
      std::string net_name;
      while(file >> net_name)
      {
        Net<float>* temp = new Net<float>(net_name, phase);
        std::cout<<"Net init done"<<std::endl;
        const std::string name = temp->name();
        nets[name] = temp;
        std::string weights = "weights/" + name + ".caffemodel";
        nets[name]->CopyTrainedLayersFrom(weights);
        std::cout<<"Weights copied done"<<std::endl;
      }
  
      reqs.push_back("imc");
      reqs.push_back("face");
      reqs.push_back("dig");
      reqs.push_back("asr");
      reqs.push_back("pos");
      reqs.push_back("ner");
      reqs.push_back("chk");
      reqs.push_back("srl");
      reqs.push_back("pt0");
      reqs.push_back("vbs");
      reqs.push_back("googlenet");
   
      // Initialize csv file lock
      if(pthread_mutex_init(&csv_lock, NULL) != 0){
        printf("Mutex init failed.\n");
        exit(1);
      }
      // Set csv file name
      csv_file_name = vm["csv"].as<string>();
  
      FILE* csv_file = fopen(csv_file_name.c_str(), "a");
  //    fprintf(csv_file, "app,plat,batch,threads,lat,qpms\n");
      fclose(csv_file);
  
      NUM_QS = vm["trial"].as<int>();
  
      int total_thread_cnt = vm["threadcnt"].as<int>();
  
      int server_sock = SERVER_init(vm["portno"].as<int>());
  
      // Listen on socket
      listen(server_sock, 10);
      LOG(INFO) << "Server is listening for requests on " << vm["portno"].as<int>();
  
      // Main Loop
      int thread_cnt = 0;
      while(1) {
          int client_sock;
          pthread_t new_thread_id;
          client_sock = accept(server_sock, (sockaddr*) 0, (unsigned int *) 0);
          if(client_sock == -1)
              printf("Failed to accept.\n");
          else {
              // Create a new thread, pass the socket number to it
              new_thread_id = request_thread_init(client_sock);
  
          }
         thread_cnt++;
         if(thread_cnt == total_thread_cnt){
           if(pthread_join(new_thread_id, NULL) != 0){
             printf("Failed to join.\n");
             exit(1);
           }
           cudaDeviceReset();
           break;
         }
      }
      // End of the server setup
    }else{
      // Local experiment setup
      string network = vm["network"].as<string>();
      string input_file = vm["input"].as<string>();
      int trial = vm["trial"].as<int>();

      LOG(INFO) << "Local inference on network " << network 
        << " with input " << input_file 
        << " for " << trial << " trials.";

      string model = "net-configs/" + network + ".prototxt"; 
      string weights = "weights/" + network + ".caffemodel";
      
      // Load in the model
      Net<float>* net = new Net<float>(model, phase);
      net->CopyTrainedLayersFrom(weights);
      LOG(INFO) << "Network initialization done w/ config: " << model << " and weights: " << weights;

      // Read in input
      // The first line in the input file should be the size of the input
      int input_size;
      ifstream file(input_file.c_str());
      if(file.is_open()){
        file >> input_size;
      }

      // Close file
      file.close();

      // Reshape the model if neccessary
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

      LOG(INFO)<<"Input size is "<< input_size <<std::endl;

      if(input_size/(c_in*w_in*h_in) != n_in){

        // Reshape input blobs
        n_in = input_size/(c_in*w_in*h_in);

        LOG(INFO) << "Reshpeing input to dims: "
          << n_in << " " << c_in << " " << w_in << " " << h_in; 
        net->input_blobs()[0]->Reshape(n_in, c_in, w_in, h_in);
        in_elts = net->input_blobs()[0]->count();

        // Reshape output blobs
        n_out = n_in;

        LOG(INFO) << "Reshpeing output to dims: "
          << n_out << " " << c_out << " " << w_out << " " << h_out; 
        net->output_blobs()[0]->Reshape(n_out, c_out, w_out, h_in);
        out_elts = net->output_blobs()[0]->count();
      }
      
      float* input = (float*) malloc(input_size * sizeof(float));
      float* output = (float*)malloc(out_elts * sizeof(float));

      // Read in the input
      for(int i = 0; i < input_size; i++){
        file >> input[i];
      }

      // Start inference
      // First a warm up pass
      float loss;
      LOG(INFO)<<"Warm up pass to move the model over";
      vector<Blob<float>* > in_blobs = net->input_blobs();
      in_blobs[0]->set_cpu_data(input);
      vector<Blob<float>* > out_blobs;
      out_blobs = net->ForwardPrefilled(&loss);
      memcpy(output, out_blobs[0]->cpu_data(), sizeof(float));
       
      // Start experiment
      LOG(INFO) << "Perform inference for " << trial << " times to average over.";
      struct timeval start, end, diff;
      float total_runtime = 0;

      if(vm["transfer"].as<bool>()){
        // Include data transfer time in timing
        LOG(INFO) << "Data transfer time included";
        gettimeofday(&start, NULL);
  
        for(int it = 0; it < trial; it++){
          input[0] += 1.0;
          in_blobs[0]->set_cpu_data(input);
         
          // Tell caffe to transfer data to GPU 
          if(vm["gpu"].as<bool>())
            in_blobs[0]->gpu_data(); 
  
          out_blobs = net->ForwardPrefilled(&loss);
          memcpy(output, out_blobs[0]->cpu_data(), sizeof(float));
        }
        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);
        total_runtime = (double)diff.tv_sec*(double)1000 
                          + (double)diff.tv_usec/(double)1000;
      }else{
        // Exclude data transfer time in timing
        LOG(INFO) << "Data transfer time excluded.";
        LOG(INFO) << "Forward only pass is being reported.";
  
        for(int it = 0; it < trial; it++){
          input[0] += 1.0;
          in_blobs[0]->set_cpu_data(input);
          
          if(vm["gpu"].as<bool>())
            in_blobs[0]->gpu_data(); // Tell caffe to ship data to GPU
  
          gettimeofday(&start, NULL);
          out_blobs = net->ForwardPrefilled(&loss);
          gettimeofday(&end, NULL);
         
          timersub(&end, &start,&diff);
          total_runtime += (double)diff.tv_sec*(double)1000 
                                + (double)diff.tv_usec/(double)1000;
        
          memcpy(output, out_blobs[0]->cpu_data(), sizeof(float));
        }
      }

    
      LOG(INFO) << "Inference done";

      if(out_elts != out_blobs[0]->count())
        LOG(FATAL) << "output size do not agree";

      // Print output result
//      LOG(INFO)<<"Printing Inference result: ";
//      for (int i = 0; i < out_elts; i++){
//        std::cout << output[i] << " ";
//      }
//      std::cout << std::endl;

      // Calculate average runtime
      float avg_runtime = total_runtime / (double)trial;

      // Write to CSV
      string csv_file = vm["csv"].as<string>();

      LOG(INFO) << "CSV file is " << csv_file;
      // First check if csv file already exist
      bool exist = ifexist(csv_file);

      ofstream csv;
      csv.open(csv_file.c_str(), ios::out | ios::app);

      if(!csv.is_open())
        LOG(FATAL) << "Can't open csv file for output.";
      
      bool verbose = vm["verbose"].as<bool>();

      LOG(INFO)<<"Verbose output: " << verbose;
      if(!exist){
        // original no csv exist
        // print header
        string header;
        if(verbose)
          header =  "net,plat,cputhread,cpufreq,runtime\n";
        else
          header = "net,plat,runtime\n"; 
        csv << header;
      }

      // Print info
      char info[100];
      if(verbose){
        sprintf(info, "%s,%s,%d,%.1f,%.4f\n",network.c_str(),
                                              platform.c_str(),
                                              openblas_threads,
                                              cpufreq,
                                              avg_runtime);
      }else{
        sprintf(info, "%s,%s,%.4f\n",network.c_str(),
                                      platform.c_str(),
                                      avg_runtime);
      }

      csv << info;

      csv.close();
      // End of local experiment setup
    }
    return 0;
}
