#ifndef _THREAD_H_
#define _THREAD_H_

#include <pthread.h>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include <vector>
#include <string>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"
#include "SENNA_VBS.h"
#include "SENNA_PT0.h"
#include "SENNA_SRL.h"

using namespace std;

enum request_type {IMC=0, FACE, DIG, ASR, POS, NER, CHK, SRL, PT0, VBS, SIZE_OF_ENUM};

static const char* request_name[SIZE_OF_ENUM] = {"imc",
                                                 "face",
                                                 "dig",
                                                 "asr",
                                                 "pos",
                                                 "ner",
                                                 "chk",
                                                 "srl",
                                                 "pt0",
                                                 "vbs"};
pthread_t request_thread_init(int sock);
void* request_handler(void* sock);
void init_mutex(void);
// Lock for the csv result
extern pthread_mutex_t csv_lock;

extern std::string csv_file_name;

extern std::string platform;
#endif // #define _THREAD_H_
