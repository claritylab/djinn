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
#include <glog/logging.h>

#include "opencv2/opencv.hpp"
#include "boost/program_options.hpp" 
#include "align.h"
#include "socket.h"

using namespace std;
using namespace cv;

namespace po = boost::program_options;

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
        ("queries,q", po::value<int>()->default_value(1), "Total number of queries to send")
        ("num,n", po::value<int>()->default_value(1), "num images (default=1)")
        ("haar,c", po::value<string>(), "(face) Haar Cascade model")
        ("flandmark,f", po::value<string>(), "(face) Flandmarks trained data")

        ("csv,c", po::value<string>(), "csv to record timing")
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
    bool make_bin = false;
    google::InitGoogleLogging(argv[0]);
    po::variables_map vm = parse_opts(argc, argv);
      /* Timing */
    struct timeval tv1, tv2, diff;
    double preproc = 0;
    double dnntime = 0;

    assert(vm.count("hostname"));

    /* open socket */
    int socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
    if(socketfd < 0)
        exit(0);

    assert(vm.count("input"));
    int NUM_IMGS = vm["num"].as<int>();
    int NUM_QS = vm["queries"].as<int>();
    
    string task = vm["task"].as<string>();
    // send req_type
    int req_type;
    // read in image
    int IMG_SIZE = 0;                         // c * w * h
    if(task == "imc") { req_type = 0; IMG_SIZE = 3 * 227 * 227; } //hardcoded for AlexNet;
    else if(task == "face") { req_type = 1; IMG_SIZE = 3 * 152 * 152; } //hardcoded for DeepFace;
    else if(task == "dig") { req_type = 2; NUM_IMGS = 100*vm["num"].as<int>(); IMG_SIZE = 1 * 28 * 28; } //hardcoded for Mnist;
    else { LOG(ERROR) << "unrecognized task."; exit(1); }

    float *arr = (float*) malloc(NUM_IMGS * IMG_SIZE * sizeof(float));
    Mat img = imread(vm["input"].as<string>().c_str(), CV_LOAD_IMAGE_COLOR);

    gettimeofday(&tv1,NULL);
    if(task == "face")
        preprocess(vm, img, NUM_IMGS);
    gettimeofday(&tv2,NULL);
    timersub(&tv2, &tv1, &diff);
    preproc = ((double)diff.tv_sec*(double)1000 + (double)diff.tv_usec/(double)1000);

    for(int n = 0; n < NUM_IMGS; ++n) {
        int img_count = 0;
        for(int c = 0; c < img.channels(); ++c) {
            for(int i = 0; i < img.rows; ++i) {
                for(int j = 0; j < img.cols; ++j) {
                    Vec3b num = img.at<Vec3b>(i,j);
                    arr[n*IMG_SIZE + img_count] = num[c];
                    ++img_count;
                }
            }
        }
    }

    SOCKET_send(socketfd, (char*)&req_type, sizeof(int), vm["debug"].as<bool>());

    // send len
    SOCKET_txsize(socketfd, NUM_IMGS * IMG_SIZE);
    float *preds = (float *) malloc(NUM_IMGS  * sizeof(float));

    for(int i = 0; i < NUM_QS; ++i) {
      // send image
      SOCKET_send(socketfd, (char*)arr, NUM_IMGS * IMG_SIZE * sizeof(float), vm["debug"].as<bool>());
      // receive data
      SOCKET_receive(socketfd, (char*)preds, NUM_IMGS * sizeof(float), vm["debug"].as<bool>());
      // check correct
      for(int j = 0; j < NUM_IMGS; ++j)
        LOG(INFO) << "Image: " << j << " class: " << preds[j] << endl;;
    }

    SOCKET_close(socketfd, false);

    free(preds);
    free(arr);

	return 0;
}
