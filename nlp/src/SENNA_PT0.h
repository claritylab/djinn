#ifndef SENNA_PT0_H
#define SENNA_PT0_H

#include "Dnn.h"
using namespace dnn;

typedef struct SENNA_PT0_
{
    /* sizes */
    int window_size;
    int ll_word_size;
    int ll_word_max_idx;
    int ll_caps_size;
    int ll_caps_max_idx;
    int ll_posl_size;
    int ll_posl_max_idx;
    int input_state_size;
    int hidden_state_size;
    int output_state_size;

    /* weights */
    double *ll_word_weight;
    double *ll_caps_weight;
    double *ll_posl_weight;
    double *l1_weight;
    double *l1_bias;
    double *l2_weight;
    double *l2_bias;
    double *viterbi_score_init;
    double *viterbi_score_trans;

    /* states */
    double *input_state;
    double *hidden_state;
    double *output_state;
    int *labels;

    /* padding indices */
    int ll_word_padding_idx;
    int ll_caps_padding_idx;
    int ll_posl_padding_idx;

    bool service;
    bool debug;

    /* internal socket */
    int socketfd;

    /* profiling */
    int calls;
    unsigned int apptime;
    unsigned int dnntime;

} SENNA_PT0;

SENNA_PT0* SENNA_PT0_new(const char *path, const char *subpath);
int* SENNA_PT0_forward(SENNA_PT0 *pt0, const int *sentence_words, const int *sentence_caps, const int *sentence_posl, int sentence_size, DnnClient client, bool service);
void SENNA_PT0_free(SENNA_PT0 *pt0);

#endif
