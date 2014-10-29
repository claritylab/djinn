/* Johann Hauswald
 * jahausw@umich.edu
 * 2014
 */

#include <assert.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <stdio.h>
#include <sys/time.h>

#include "boost/program_options.hpp" 
#include "caffe/caffe.hpp"
#include "align.h"
#include "socket.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

using namespace std;

namespace po = boost::program_options;

#define TIMING 1

po::variables_map parse_opts( int ac, char** av )
{
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "Produce help message")
        ("hostname,h", po::value<string>(), "Server IP addr")
        ("portno,p", po::value<int>()->default_value(8080), "Server port (default: 8080)")
        ("task,t", po::value<string>(), "Image task: imc (ImageNet), face (DeepFace)")
        ("imcin,i", po::value<string>(), "Mini net for input")
        ("haar,c", po::value<string>(), "(face) Haar Cascade model")
        ("flandmark,f", po::value<string>(), "(face) Flandmarks trained data")

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

int main( int argc, char** argv )
{
	po::variables_map vm = parse_opts(argc, argv);
    /* Timing */
	struct timeval tv1, tv2;
    unsigned int apptime = 0;
    unsigned int txtime = 0;

    assert(vm.count("hostname"));

    /* open socket */
    int socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
    if(socketfd < 0)
        exit(0);

    gettimeofday(&tv1,NULL);
    Caffe::set_phase(Caffe::TEST);
    if(vm["gpu"].as<bool>())
        Caffe::set_mode(Caffe::GPU);
    else
        Caffe::set_mode(Caffe::CPU);

    float loss;
    assert(vm.count("imcin"));
    Net<float>* espresso = new Net<float>(vm["imcin"].as<string>());
    const caffe::LayerParameter& in_params = espresso->layers()[0]->layer_param();

    // preprocess face outside of NN for facial recognition before forward pass which loads image(s)
    // $cmt: still tries to do facial recognition if no faces or landmarks found.
    string task = vm["task"].as<string>();
    if(task == "face")
        if(preprocess(vm, in_params.image_data_param().source()) == false)
            exit(0);

    gettimeofday(&tv2,NULL);
    apptime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);
    
    // not timed disk I/O stuff
    vector<Blob<float>* > img_blobs = espresso->ForwardPrefilled(&loss);

    gettimeofday(&tv1,NULL);
    // send req_type
    int req_type;
    if(task == "imc") req_type = 0;
    else if(task == "face") req_type = 1;
    else if(task == "digit") req_type = 2;

    SOCKET_send(socketfd, (char*)&req_type, sizeof(int), vm["debug"].as<bool>());

    // send len
    SOCKET_txsize(socketfd, img_blobs[0]->count());

    // send image
    SOCKET_send(socketfd, (char*)img_blobs[0]->cpu_data(), img_blobs[0]->count()*sizeof(float), vm["debug"].as<bool>());

    // receive data
    vector<Blob<float>* > in_blobs = espresso->output_blobs();
    float *preds = (float *) malloc(in_blobs[0]->num() * sizeof(float));
    SOCKET_receive(socketfd, (char*)preds, in_blobs[0]->num() * sizeof(float), vm["debug"].as<bool>());

    gettimeofday(&tv2,NULL);
    txtime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);

    gettimeofday(&tv1,NULL);
    // check correct
    for(int j = 0; j < in_blobs[0]->num(); ++j)
        cout << "Image: " << j << " class: " << preds[j] << endl;;

    SOCKET_close(socketfd, false);

    free(preds);
    gettimeofday(&tv2,NULL);
    apptime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);
#ifdef TIMING
    cout << "TIMING:" << endl;
    cout << "task " << task
         << " size_kb " << (float)(img_blobs[0]->count()*sizeof(float))/1024
         << " total_t " << (float)(apptime+txtime)/1000
         << " app_t " << (float)(apptime/1000)
         << " tx_t " << (float)(txtime/1000) << endl;
#endif
    
	return 0;
}
