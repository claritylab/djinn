#ifndef __ALIGN_H_
#define __ALIGN_H_

#include <string>
#include "boost/program_options.hpp"
#include "opencv2/opencv.hpp"

namespace po = boost::program_options;

void preprocess(po::variables_map& vm, cv::Mat image, int NUM_IMGS);

#endif
