#ifndef _THREAD_H_
#define _THREAD_H_

#include <pthread.h>
#include <stdio.h>
#include "dnn.h"
#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"
#include "SENNA_VBS.h"
#include "SENNA_PT0.h"
#include "SENNA_SRL.h"

using namespace std;

enum request_type {IMC=0, ASR, POS, NER, CHK, SRL, PT0, VBS, FACE, DIG, SIZE_OF_ENUM};
static const char* request_name[SIZE_OF_ENUM] = {"imc", "asr", "pos", "ner", "chk", "srl", "pt0", "vbs",
                                                "face", "dig"};
int request_thread_init(int sock);
void* request_handler(void* sock);


#endif // #define _THREAD_H_
