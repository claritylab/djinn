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
#include "dnn.h"
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
    
    int server_sock = SERVER_init(vm["portno"].as<int>());
    
    // Listen on socket
    listen(server_sock, 10);
    printf("Server is listening for request on %d\n", 8080);

    Caffe::set_phase(Caffe::TEST);
    if(vm["gpu"].as<bool>()){
      Caffe::set_mode(Caffe::GPU);
      printf("Going to use GPU\n");
    }else{
      Caffe::set_mode(Caffe::CPU);
      printf("Going to use CPU\n");
    }
    // Main Loop
    while(1){
      int client_sock;
      client_sock = accept(server_sock, (sockaddr*) 0, (unsigned int *) 0);
      if(client_sock == -1){
        printf("Failed to accept.\n");
      }else{
        // Create a new thread, pass the socket number to it
        if(request_thread_init(client_sock) == -1){
          printf("Failed to accept.\n");
        }
      }
    }
    return 0;
}
