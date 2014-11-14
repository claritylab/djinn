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
        ("task,t", po::value<string>(), "Image task: imc (ImageNet), face (DeepFace), dig (LeNet)")
        ("input,i", po::value<string>(), "input image (.bin)")
        ("num,n", po::value<int>()->default_value(1), "num images (default=1)")
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

    assert(vm.count("input"));
    int NUM_IMGS = vm["num"].as<int>();

    gettimeofday(&tv1,NULL);

    gettimeofday(&tv2,NULL);
    apptime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);
    
    string task = vm["task"].as<string>();
    // send req_type
    int req_type;
    // read in image
    int IMG_SIZE = 0;                          // c * w * h
    if(task == "imc") { req_type = 0; IMG_SIZE = 3 * 227 * 227; } //hardcoded for AlexNet;
    else if(task == "face") { req_type = 1; IMG_SIZE = 3 * 152 * 152; } //hardcoded for DeepFace;
    else if(task == "dig") { req_type = 2; NUM_IMGS = 100; IMG_SIZE = 1 * 28 * 28; } //hardcoded for Mnist;
    else { printf("unrecognized task.\n"); exit(1); }

    float *arr = (float*) malloc(NUM_IMGS * IMG_SIZE * sizeof(float));
    std::ifstream img(vm["input"].as<string>().c_str(), std::ios::binary);
    for(int i = 0; i < NUM_IMGS * IMG_SIZE; ++i)
        img.read((char*)&(arr[i]), sizeof(float));

    // preprocess face outside of NN for facial recognition before forward pass which loads image(s)
    // $cmt: still tries to do facial recognition if no faces or landmarks found.
    if(task == "face")
        if(preprocess(vm, arr) == false)
            exit(0);

    gettimeofday(&tv1,NULL);
    SOCKET_send(socketfd, (char*)&req_type, sizeof(int), vm["debug"].as<bool>());

    // send len
    SOCKET_txsize(socketfd, NUM_IMGS * IMG_SIZE);

    // send image
    SOCKET_send(socketfd, (char*)arr, NUM_IMGS * IMG_SIZE * sizeof(float), vm["debug"].as<bool>());

    // receive data
    float *preds = (float *) malloc(NUM_IMGS  * sizeof(float));
    SOCKET_receive(socketfd, (char*)preds, NUM_IMGS * sizeof(float), vm["debug"].as<bool>());

    gettimeofday(&tv2,NULL);
    txtime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);

    // check correct
    for(int j = 0; j < NUM_IMGS; ++j)
        cout << "Image: " << j << " class: " << preds[j] << endl;;

    SOCKET_close(socketfd, false);

    free(preds);
    free(arr);
#ifdef TIMING
    cout << "TIMING:" << endl;
    cout << "task " << task
         << " size_kb " << (float)(IMG_SIZE*sizeof(float))/1024
         << " total_t " << (float)(apptime+txtime)/1000
         << " app_t " << (float)(apptime/1000)
         << " tx_t " << (float)(txtime/1000) << endl;
#endif
    
	return 0;
}
