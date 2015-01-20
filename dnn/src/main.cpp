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

std::vector<std::string> reqs;
map<string, Net<float>* > nets;
// float *in;
// float *out;
int NUM_QS;

po::variables_map parse_opts( int ac, char** av )
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")

        ("portno,p", po::value<int>()->default_value(8080), "Open server on port (default: 8080)")

        ("gpu,u", po::value<bool>()->default_value(false), "Use GPU?")
        ("debug,v", po::value<bool>()->default_value(false), "Turn on all debug")
        ("threadcnt,t", po::value<int>()->default_value(0), "Number of threads to spawn before exiting the server.")
        ("queries,q", po::value<int>()->default_value(1), "Total num queries (default: 1)")
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
    Caffe::set_phase(Caffe::TEST);
    if(vm["gpu"].as<bool>())
        Caffe::set_mode(Caffe::GPU);
    else
        Caffe::set_mode(Caffe::CPU);

    // TODO: hacky way to hold load all, find better
    std::ifstream file ("nets.txt");
    std::string net_name;
    while(file >> net_name)
    {
      Net<float>* temp = new Net<float>(net_name);
      const std::string name = temp->name();
      nets[name] = temp;
      std::string weights = "weights/" + name + ".caffemodel";
      nets[name]->CopyTrainedLayersFrom(weights);
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
 
    NUM_QS = vm["queries"].as<int>();

    // how many threads needed for a given application
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
        if(thread_cnt == total_thread_cnt) {
            if(pthread_join(new_thread_id, NULL) != 0) {
                printf("Failed to join.\n");
                exit(1);
            }
            cudaDeviceReset();
            break;
        }
    }
    return 0;
}
