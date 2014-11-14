#ifndef __ALIGN_H_
#define __ALIGN_H_

#include <string>
#include "boost/program_options.hpp"

namespace po = boost::program_options;

bool preprocess(po::variables_map& vm, float *data);

#endif
