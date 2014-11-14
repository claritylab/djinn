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

#include "boost/program_options.hpp" 
#include "socket.h"
#include "thread.h"

using namespace std;
namespace po = boost::program_options;

po::variables_map parse_opts( int ac, char** av )
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")

        ("portno,p", po::value<int>()->default_value(8080), "Open server on port (default: 8080)")

        ("gpu,u", po::value<bool>()->default_value(false), "Use GPU?")
        ("debug,v", po::value<bool>()->default_value(false), "Turn on all debug")
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
    // Main thread for the server
    // Spawn a new thread for each request
    po::variables_map vm = parse_opts(argc, argv);
 
    Caffe::SetDevice(0);

    Caffe::set_phase(Caffe::TEST);
    std::cout << "3" << std::endl;
    if(vm["gpu"].as<bool>()){
        Caffe::set_mode(Caffe::GPU);
    }
    else
        Caffe::set_mode(Caffe::CPU);

    int server_sock = SERVER_init(vm["portno"].as<int>());

    // Listen on socket
    listen(server_sock, 10);
    printf("Server is listening for request on %d\n", vm["portno"].as<int>());

    init_mutex();

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
        /*
        if(thread_cnt == 2){
          if(pthread_join(new_thread_id, NULL) != 0){
            printf("Failed to join.\n");
            exit(1);
          }
          cudaDeviceReset();
          break;
        }*/
    }
    return 0;
}
