/* Johann Hauswald, Yiping Kang
 * {jahausw, ypkang}@umich.edu
 * 2014
 */

#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <unistd.h>  
#include <errno.h>
#include <map>
#include <glog/logging.h>
#include <sys/stat.h>

#include "boost/program_options.hpp" 

#include "caffe/caffe.hpp"

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
        ("gpu,u", po::value<bool>()->default_value(true), "Use GPU?")
        ("debug,v", po::value<bool>()->default_value(false), "Turn on all debug")

        // Options for local setup
        ("network,n", po::value<string>()->default_value("dummy.prototxt"), "DNN network to use in this experiment")
        ("input,i", po::value<string>()->default_value("undefined"), "Input to the DNN")
        ("trial,r", po::value<int>()->default_value("1"), "number of trials")
        ("threads,t", po::value<int>()->default_value(1), "CPU threads used (default: 0)")
        ("layer_csv,l", po::value<string>()->default_value("NO_LAYER"), "CSV file to put layer latencies in.")
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
    caffe::Phase phase = 1; // Set phase to TEST
    int openblas_threads = 1;

    if(vm["gpu"].as<bool>()){
        Caffe::set_mode(Caffe::GPU);
    }
    else{
        Caffe::set_mode(Caffe::CPU);
        openblas_threads = vm["threads"].as<int>();
    }

    vector<Blob<float>*> bottom;
    Net<float>* net = new Net<float>(vm["network"].as<string>().c_str(), phase);
    float loss;
    net->ForwardPrefilled(&loss);
    
    for(int i = 0; i < vm["trial"].as<int>(); i++)
      net->ForwardPrefilled(&loss, vm["layer_csv"].as<string>());
  
    return 0;
}
