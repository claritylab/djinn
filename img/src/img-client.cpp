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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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
    google::InitGoogleLogging(argv[0]);
    po::variables_map vm = parse_opts(argc, argv);
      /* Timing */
    struct timeval tp1, tp2, tv1, tv2;
    unsigned int apptime = 0;
    unsigned int txtime = 0;
    unsigned int throughput = 0;

    assert(vm.count("hostname"));

    /* open socket */
    int socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
    if(socketfd < 0)
        exit(0);

    assert(vm.count("input"));
    int NUM_IMGS = vm["num"].as<int>();
    int NUM_QS = vm["queries"].as<int>();

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
    else if(task == "dig") { req_type = 2; NUM_IMGS = 100*vm["num"].as<int>(); IMG_SIZE = 1 * 28 * 28; } //hardcoded for Mnist;
    else if(task == "googlenet") { req_type = 10; IMG_SIZE = 3*224*224; }
    else { LOG(ERROR) << "unrecognized task."; exit(1); }

    // Read in image using opencv
    string input_image = vm["input"].as<string>();
    std::cout<<"input file is "<<input_image<<std::endl;
    Mat image;
    if(task == "dig")
      image = imread(vm["input"].as<string>().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    else
      image = imread(vm["input"].as<string>().c_str(), CV_LOAD_IMAGE_COLOR);
    if(! image.data )
      std::cout<<"Could not open the image"<<std::endl;
//    unsigned char *arr_char = (unsigned char*) malloc(NUM_IMGS * IMG_SIZE * sizeof(unsigned char));
//    std::ifstream img(vm["input"].as<string>().c_str(), std::ios::binary | std::ios::in);
//    for(int j = 0; j < NUM_IMGS; ++j) {
//      for(int i = 0; i < IMG_SIZE; ++i) {
//          img.read(arr_char[j*IMG_SIZE + i], sizeof(unsigned char));
//          img.seekg(ios::beg);
//      }
//    }

    std::cout<<"Dimension"<<image.rows << " "<<image.cols<<std::endl;
    // Conver into floating point
    float *arr = (float*) malloc(NUM_IMGS * IMG_SIZE * sizeof(float));
    for(int i = 0; i < IMG_SIZE; i++){
      arr[i] = (float)(image.data[i]); 
      std::cout<<arr[i] << " " << (int)(image.data[i]) << std::endl;
      std::cout<<i<<std::endl;
    }
    
    // Dump input file
    bool dump_input = true;
    if(dump_input){
      std::ofstream input_file;
      input_file.open("googlenet.in", std::ios::out);
      input_file << NUM_IMGS * IMG_SIZE << "\n"; 
      for(int j = 0; j < NUM_IMGS; j++){
        for(int i = 0; i < IMG_SIZE; i++){
          input_file << arr[i] << " ";
        }
      }
      input_file.close();
    }

    // TODO: fix
    // preprocess face outside of NN for facial recognition before forward pass which loads image(s)
    // $cmt: still tries to do facial recognition if no faces or landmarks found.
    // if(task == "face")
    //     preprocess(vm, arr, NUM_IMGS);

    SOCKET_send(socketfd, (char*)&req_type, sizeof(int), vm["debug"].as<bool>());

    // send len
    SOCKET_txsize(socketfd, NUM_IMGS * IMG_SIZE);
    float *preds = (float *) malloc(NUM_IMGS  * sizeof(float));

    gettimeofday(&tp1,NULL);
    for(int i = 0; i < NUM_QS; ++i) {
      // send image
      gettimeofday(&tv1,NULL);
      SOCKET_send(socketfd, (char*)arr, NUM_IMGS * IMG_SIZE * sizeof(float), vm["debug"].as<bool>());
      // receive data
      SOCKET_receive(socketfd, (char*)preds, NUM_IMGS * sizeof(float), vm["debug"].as<bool>());
      gettimeofday(&tv2,NULL);
      txtime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);
      // check correct
      for(int j = 0; j < NUM_IMGS; ++j)
        LOG(INFO) << "Image: " << j << " class: " << preds[j] << endl;;
    }
    gettimeofday(&tp2,NULL);
    throughput = (tp2.tv_sec-tp1.tv_sec)*1000000 + (tp2.tv_usec-tp1.tv_usec);

    SOCKET_close(socketfd, false);

    free(preds);
    free(arr);

    // Set csv file name
    if(vm.count("csv")) {
      string csv_file_name = vm["csv"].as<string>();
      FILE* csv_file = fopen(csv_file_name.c_str(), "a");
      fprintf(csv_file, "task,qs,data,totaltime,app,transfer\n");
      // 
      fprintf(csv_file, "%s,%d,%.2f,%.2f,%.2f,%.2f\n", task.c_str(),
                                                NUM_QS,
                                                (float)(NUM_IMGS * IMG_SIZE*sizeof(float))/1024,
                                                (float)(apptime+txtime)/1000,
                                                (float)(apptime/1000),
                                                (float)(txtime/1000)
                                                );
      fclose(csv_file);
    }
    
	return 0;
}
