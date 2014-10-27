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

    assert(vm.count("hostname"));

    /* open socket */
    int socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
    if(socketfd < 0)
        exit(0);

    Caffe::set_phase(Caffe::TEST);
    if(vm["gpu"].as<bool>())
        Caffe::set_mode(Caffe::GPU);
    else
        Caffe::set_mode(Caffe::CPU);

    float loss;
    assert(vm.count("imcin"));
    Net<float>* espresso = new Net<float>(vm["imcin"].as<string>());
    const caffe::LayerParameter& in_params = espresso->layers()[0]->layer_param();
    ifstream img_list ();

    // preprocess face outside of NN for facial recognition before forward pass which loads image(s)
    // $cmt: still tries to do facial recognition if no faces or landmarks found.
    if(vm["task"].as<string>() == "face")
        if(preprocess(vm, in_params.image_data_param().source()) == false)
            exit(0);

    vector<Blob<float>* > img_blobs = espresso->ForwardPrefilled(&loss);
    // send size
    SOCKET_txsize(socketfd, img_blobs[0]->count());

    // send image
    SOCKET_send(socketfd, (char*)img_blobs[0]->cpu_data(), img_blobs[0]->count()*sizeof(float), vm["debug"].as<bool>());

    // receive data
    vector<Blob<float>* > in_blobs = espresso->output_blobs();
    float *preds = (float *) malloc(in_blobs[0]->num() * sizeof(float));
    SOCKET_receive(socketfd, (char*)preds, in_blobs[0]->num() * sizeof(float), vm["debug"].as<bool>());

    // check correct
    for(int j = 0; j < in_blobs[0]->num(); ++j)
        LOG(INFO) << " Image: " << j << " class: " << preds[j];

    SOCKET_close(socketfd, false);

    free(preds);
    
	return 0;
}
