#ifndef SENNA_SRL_H
#define SENNA_SRL_H

#include "Dnn.h"
using namespace dnn;

typedef struct SENNA_SRL_
{
    /* sizes */
    int window_size;
    int ll_word_size;
    int ll_word_max_idx;
    int ll_caps_size;
    int ll_caps_max_idx;
    int ll_chkl_size;
    int ll_chkl_max_idx;
    int ll_posv_size;
    int ll_posv_max_idx;
    int ll_posw_size;
    int ll_posw_max_idx;
    int input_state_size;
    int hidden_state1_size;
    int hidden_state3_size;
    int output_state_size;

    /* weights */
    double *ll_word_weight;
    double *ll_caps_weight;
    double *ll_chkl_weight;
    double *ll_posv_weight;
    double *ll_posw_weight;
    double *l1_weight_wcc;
    double *l1_weight_pw;
    double *l1_weight_pv;
    double *l1_bias;
    double *l3_weight;
    double *l3_bias;
    double *l4_weight;
    double *l4_bias;
    double *viterbi_score_init;
    double *viterbi_score_trans;

    /* extra inputs */
    int *sentence_posv;    
    int *sentence_posw;

    /* states */
    double *input_state;
    double *input_state_wcc;
    double *input_state_pw;
    double *input_state_pv;
    double *hidden_state1;
    double *hidden_state1_wcc;
    double *hidden_state1_pw;
    double *hidden_state1_pv;
    double *hidden_state2;
    double *hidden_state3;
    double *output_state;
    int **labels;
    int labels_size;

    /* padding indices */
    int ll_word_padding_idx;
    int ll_caps_padding_idx;
    int ll_chkl_padding_idx;

    /* service flag */
    bool service;
    bool debug;

    /* profiling */
    int calls;
    unsigned int apptime;
    unsigned int dnntime;

} SENNA_SRL;

SENNA_SRL* SENNA_SRL_new(const char *path, const char *subpath);
int** SENNA_SRL_forward(SENNA_SRL *srl, const int *sentence_words, const int *sentence_caps, const int *sentence_chkl, const int *sentence_isvb, int sentence_size, DnnClient client, bool service);
void SENNA_SRL_free(SENNA_SRL *srl);

#endif
