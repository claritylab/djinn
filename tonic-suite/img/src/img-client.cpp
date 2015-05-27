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

#include "opencv2/opencv.hpp"
#include "boost/program_options.hpp" 
#include "caffe/caffe.hpp"
#include "align.h"
#include "socket.h"
#include "tonic.h"

using namespace std;
using namespace cv;

namespace po = boost::program_options;
struct timeval tv1, tv2;

po::variables_map parse_opts( int ac, char** av )
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message")
    ("task,t", po::value<string>()->default_value("imc"), "Image task: imc (ImageNet), face (DeepFace), dig (LeNet)")
    ("network,n", po::value<string>()->default_value("model/imc.prototxt"), "Network config file (.prototxt)")
    ("model,m", po::value<string>()->default_value("model/imc.caffemodel"), "Pretrained weights (.caffemodel)")
    ("input,i", po::value<string>()->default_value("input/imc-list.txt"), "List of ingput images (1 jpg/line)")
    ("num,u", po::value<int>()->default_value(1), "num images to read from input list file")

    ("djinn,d", po::value<bool>()->default_value(false), "Use DjiNN service?")
    ("hostname,o", po::value<string>()->default_value("localhost"), "Server IP addr")
    ("portno,p", po::value<int>()->default_value(8080), "Server port")

  // facial recognition flags
    ("align,l", po::value<bool>()->default_value(true), "(face) align images before inference?")
    ("haar,a", po::value<string>()->default_value("data/haar.xml"), "(face) Haar Cascade model")
    ("flandmark,f", po::value<string>()->default_value("data/flandmark.dat"), "(face) Flandmarks trained data")

    ("gpu,g", po::value<bool>()->default_value(false), "Use GPU?")
    ("debug,v", po::value<bool>()->default_value(false), "Turn on all debug") 
    ;

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
		cout << desc << "\n";
        exit(1);
	}
	return vm;
}

int main( int argc, char** argv )
{
  po::variables_map vm = parse_opts(argc, argv);

  bool debug = vm["debug"].as<bool>();

  TonicSuiteApp app;
  app.task = vm["task"].as<string>();
  app.network = vm["network"].as<string>();
  app.model = vm["model"].as<string>();
  app.input = vm["input"].as<string>();
  app.num_imgs = vm["num"].as<int>();

  // DjiNN service or local?
  app.djinn = vm["djinn"].as<bool>();
  app.gpu = vm["gpu"].as<bool>();

  if(app.djinn) {
    app.hostname = vm["hostname"].as<string>();
    app.portno = vm["portno"].as<string>();
    app.socketfd = CLIENT_init(app.hostname.c_str(), app.portno.c_str(), debug);
    if(app.socketfd < 0)
      exit(0);
  }
  else {
      app.net = new Net<float>(app.network);
      app.net->CopyTrainedLayersFrom(app.model);
  }

  // send req_type
  // FIXME: req type
  int req_type;
  // read in image
  app.img_size = 0;                          // c * w * h
  //hardcoded for AlexNet
  if(app.task == "imc") { req_type = 0; app.img_size = 3 * 227 * 227; } 
  //hardcoded for DeepFace
  else if(app.task == "face") { req_type = 1; app.img_size = 3 * 152 * 152; }
  //hardcoded for Mnist
  else if(app.task == "dig") { req_type = 2; app.img_size = 1 * 28 * 28; }
  else { printf("unrecognized task.\n"); exit(1); }

  map<string, Mat> imgs;
  std::ifstream file (app.input.c_str());
  std::string img_file;
  int skips = 0;
  for(int i = 0; i < app.num_imgs; ++i) {
    file >> img_file;
    LOG(INFO) << "Reading " << img_file;
    Mat img;
    if(app.task == "dig")
      img = imread(img_file, CV_LOAD_IMAGE_GRAYSCALE);
    else
      img = imread(img_file);

    if(img.channels()*img.rows*img.cols != app.img_size) {
      LOG(ERROR) << "Skipping " << img_file << ", resize to correct dimensions.\n";
      ++skips;
    }
    else
      imgs[img_file] = img;
  }
  // remove skipped images
  app.num_imgs -= skips;

  map<string, Mat>::iterator it;
  // align facial recognition image
  if(app.task == "face" && vm["align"].as<bool>()) {
    for(it = imgs.begin(); it != imgs.end(); ++it) {
      LOG(INFO) << "Aligning: " << it->first << endl;
      preprocess(it->second, vm["flandmark"].as<string>(), vm["haar"].as<string>());
      imwrite(it->first, it->second);
    }
  }

  // prepare data into array
  float *arr = (float*) malloc(app.num_imgs * app.img_size * sizeof(float));
  float *preds = (float*) malloc(app.num_imgs  * sizeof(float));

  int img_count = 0;
  for(it = imgs.begin(); it != imgs.end(); ++it) {
    int pix_count = 0;
    for(int c = 0; c < it->second.channels(); ++c) {
      for(int i = 0; i < it->second.rows; ++i) {
        for(int j = 0; j < it->second.cols; ++j) {
          Vec3b pix = it->second.at<Vec3b>(i,j);
          arr[img_count*app.img_size + pix_count] = pix[c];
          ++pix_count;
        }
      }
    }
    ++img_count;
  }

  if(app.djinn) {
    SOCKET_send(app.socketfd, (char*)&req_type, sizeof(int), debug);

    // send len
    SOCKET_txsize(app.socketfd, app.num_imgs * app.img_size);

    // send image(s)
    SOCKET_send(app.socketfd, (char*)arr, app.num_imgs * app.img_size * sizeof(float), debug);

    SOCKET_receive(app.socketfd, (char*)preds, app.num_imgs * sizeof(float), debug);
    SOCKET_close(app.socketfd, debug);
  }
  else {
    reshape(app.net, app.num_imgs * app.img_size);
    float loss;
    vector<Blob<float>* > in_blobs = app.net->input_blobs();

    in_blobs[0]->set_cpu_data(arr);
    vector<Blob<float>* > out_blobs = app.net->ForwardPrefilled(&loss);
    memcpy(preds, out_blobs[0]->cpu_data(), app.num_imgs*sizeof(float));
  }

  for(it = imgs.begin(); it != imgs.end(); it++) {
    LOG(INFO) << "Image: " << it->first << " class: " << preds[distance(imgs.begin(), it)] << endl;;
  }

  free(preds);
  free(arr);
  free(app.net);

  return 0;
}
