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

using namespace std;
namespace po = boost::program_options;

po::variables_map parse_opts( int ac, char** av )
{
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "Produce help message")

        ("portno,p", po::value<int>()->default_value(8080), "Open server on port (default: 8080)")
        ("service,s", po::value<string>(), "Service type: imc, pos")
        ("model,m", po::value<string>(), "Model definition file (.prototxt)")
        ("weights,w", po::value<string>(), "Trained model weights (.caffemodel, .dat)") 

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
	po::variables_map vm = parse_opts(argc, argv);

    Caffe::set_phase(Caffe::TEST);
    if(vm["gpu"].as<bool>())
        Caffe::set_mode(Caffe::GPU);
    else
        Caffe::set_mode(Caffe::CPU);

    Net<float>* espresso = SERVICE_init(vm);

    int portno = vm["portno"].as<int>();
    printf("Listening for %s requests on %d\n...", vm["service"].as<string>().c_str(), portno);
    int socketfd = SERVER_init(portno);

    // rx len
    int sock_elts = SOCKET_rxsize(socketfd);
    if(sock_elts < 0) {
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

    // beast mode
    while(1)
    {
        int rcvd = SOCKET_receive(socketfd, (char*) in, in_elts*sizeof(float), vm["debug"].as<bool>());
        if(rcvd == 0) continue;
        SERVICE_fwd(in, in_elts, out, out_elts, espresso);
        SOCKET_send(socketfd, (char*) out, out_elts*sizeof(float), vm["debug"].as<bool>());
    }

    SOCKET_close(socketfd, vm["debug"].as<bool>());
    SERVICE_close();

    free(in);
    free(out);
    
    return 0;
}
