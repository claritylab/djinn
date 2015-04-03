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

        ("portno,p", po::value<int>()->default_value(8080), "Open server on port (default: 8080)")

        ("gpu,u", po::value<bool>()->default_value(false), "Use GPU?")
        ("debug,v", po::value<bool>()->default_value(false), "Turn on all debug")
        ("csv,c", po::value<string>()->default_value("./timing.csv"), "CSV file to put the timings in.")
        ("threadcnt,t", po::value<int>()->default_value(0), "Number of threads to spawn before exiting the server.")
        ("queries,q", po::value<int>()->default_value(1), "Total num queries (default: 1)")
        ("cputhread,c", po::value<int>()->default_value(1), "Number of threads openblas uses (default :1)")
        ("cpufreq,f", po::value<int>()->default_value(2320500), "The cpu frequency (default: 2.3GHz)")
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

int main(int argc , char *argv[])
{
    // google::InitGoogleLogging(argv[0]);

    // Main thread for the server
    // Spawn a new thread for each request
    po::variables_map vm = parse_opts(argc, argv);
    caffe::Phase phase = 1;
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

    NUM_QS = vm["queries"].as<int>();

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
    return 0;
}
