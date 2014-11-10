#ifndef SENNA_POS_H
#define SENNA_POS_H

#include "../../gen-cpp2/Dnn.h"
using namespace 

struct SENNA_POS
{
    /* sizes */
    int window_size;
    int ll_word_size;
    int ll_word_max_idx;
    int ll_caps_size;
    int ll_caps_max_idx;
    int ll_suff_size;
    int ll_suff_max_idx;
    int input_state_size;
    int hidden_state_size;
    int output_state_size;

    /* weights */
    double *ll_word_weight;
    double *ll_caps_weight;
    double *ll_suff_weight;
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
    int ll_suff_padding_idx;

    /* service flag */
    bool service;
    bool debug;

    /* internal socket */
    int socketfd;

    /* profiling */
    int calls;
    unsigned int apptime;
    unsigned int dnntime;

} ;

SENNA_POS* SENNA_POS_new(const char *path, const char *subpath);
int* SENNA_POS_forward(SENNA_POS *pos, const int *sentence_words, const int *sentence_caps, const int *sentence_suff, int sentence_size, DnnAsyncClient client, Work input);
void SENNA_POS_free(SENNA_POS *pos);

#endif
