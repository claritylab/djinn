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

enum request_type {IMC, ASR, POS, NER, CHK, SRL, PT0, VBS, FACE, DIG};
int request_thread_init(int sock);
void* request_handler(void* sock);


#endif // #define _THREAD_H_
