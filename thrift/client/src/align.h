#ifndef __ALIGN_H_
#define __ALIGN_H_

#include "opencv2/opencv.hpp"
#include "flandmark_detector.h"
#include "linreg.h"

using namespace cv;

void preprocess(CvMat* img, CascadeClassifier* face_cascade, FLANDMARK_Model* model);

#endif
