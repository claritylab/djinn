#include "SENNA_SRL.h"
#include "SENNA_utils.h"

SENNA_SRL* SENNA_SRL_new(const char *path)
{
  SENNA_SRL *srl = SENNA_malloc(sizeof(SENNA_SRL), 1);
  FILE *f;
  float dummy;
  int dummy_size;

  f = SENNA_fopen(path, "rb");

  SENNA_fread(&srl->window_size, sizeof(int), 1, f);
  SENNA_fread_tensor_2d(&srl->ll_word_weight, &srl->ll_word_size, &srl->ll_word_max_idx, f);
  SENNA_fread_tensor_2d(&srl->ll_caps_weight, &srl->ll_caps_size, &srl->ll_caps_max_idx, f);
  SENNA_fread_tensor_2d(&srl->ll_chkl_weight, &srl->ll_chkl_size, &srl->ll_chkl_max_idx, f);
  SENNA_fread_tensor_2d(&srl->ll_posv_weight, &srl->ll_posv_size, &srl->ll_posv_max_idx, f);
  SENNA_fread_tensor_2d(&srl->ll_posw_weight, &srl->ll_posw_size, &srl->ll_posw_max_idx, f);
  SENNA_fread_tensor_2d(&srl->l1_weight_wcc, &dummy_size, &srl->hidden_state1_size, f);
  SENNA_fread_tensor_2d(&srl->l1_weight_pv, &dummy_size, &srl->hidden_state1_size, f);
  SENNA_fread_tensor_2d(&srl->l1_weight_pw, &dummy_size, &srl->hidden_state1_size, f);
  SENNA_fread_tensor_1d(&srl->l1_bias, &srl->hidden_state1_size, f);
  SENNA_fread_tensor_2d(&srl->l3_weight, &srl->hidden_state1_size, &srl->hidden_state3_size, f);
  SENNA_fread_tensor_1d(&srl->l3_bias, &srl->hidden_state3_size, f);
  SENNA_fread_tensor_2d(&srl->l4_weight, &srl->hidden_state3_size, &srl->output_state_size, f);
  SENNA_fread_tensor_1d(&srl->l4_bias, &srl->output_state_size, f);
  SENNA_fread_tensor_1d(&srl->viterbi_score_init, &srl->output_state_size, f);
  SENNA_fread_tensor_2d(&srl->viterbi_score_trans, &srl->output_state_size, &srl->output_state_size, f);

  SENNA_fread(&srl->ll_word_padding_idx, sizeof(int), 1, f);
  SENNA_fread(&srl->ll_caps_padding_idx, sizeof(int), 1, f);
  SENNA_fread(&srl->ll_chkl_padding_idx, sizeof(int), 1, f);

  SENNA_fread(&dummy, sizeof(float), 1, f);
  SENNA_fclose(f);

  if((int)dummy != 777)
    SENNA_error("srl: data corrupted (or not IEEE floating computer)");

  /* states */
  srl->sentence_posv = NULL;    
  srl->sentence_posw = NULL;
  srl->input_state = NULL;
  srl->input_state_wcc = NULL;
  srl->input_state_pv = NULL;
  srl->input_state_pw = NULL;
  srl->hidden_state1 = NULL;
  srl->hidden_state1_wcc = NULL;
  srl->hidden_state1_pv = NULL;
  srl->hidden_state1_pw = NULL;
  srl->hidden_state2 = NULL;
  srl->hidden_state3 = NULL;
  srl->output_state = NULL;
  srl->labels = NULL;
  srl->labels_size = 0;

  /* some info if you want verbose */
  SENNA_message("srl: window size: %d", srl->window_size);
  SENNA_message("srl: vector size in word lookup table: %d", srl->ll_word_size);
  SENNA_message("srl: word lookup table size: %d", srl->ll_word_max_idx);
  SENNA_message("srl: vector size in caps lookup table: %d", srl->ll_caps_size);
  SENNA_message("srl: caps lookup table size: %d", srl->ll_caps_max_idx);
  SENNA_message("srl: vector size in verb position lookup table: %d", srl->ll_posv_size);
  SENNA_message("srl: verb position lookup table size: %d", srl->ll_posv_max_idx);
  SENNA_message("srl: vector size in word position lookup table: %d", srl->ll_posw_size);
  SENNA_message("srl: word position lookup table size: %d", srl->ll_posw_max_idx);
  SENNA_message("srl: number of hidden units (convolution): %d", srl->hidden_state1_size);
  SENNA_message("srl: number of hidden units (hidden layer): %d", srl->hidden_state3_size);
  SENNA_message("srl: number of classes: %d", srl->output_state_size);

  return srl;
}

void SENNA_SRL_free(SENNA_SRL *srl)
{
  int i;

  /* weights */
  SENNA_free(srl->ll_word_weight);
  SENNA_free(srl->ll_caps_weight);
  SENNA_free(srl->ll_chkl_weight);
  SENNA_free(srl->ll_posv_weight);
  SENNA_free(srl->ll_posw_weight);
  SENNA_free(srl->l1_weight_wcc);
  SENNA_free(srl->l1_weight_pv);
  SENNA_free(srl->l1_weight_pw);
  SENNA_free(srl->l1_bias);
  SENNA_free(srl->l3_weight);
  SENNA_free(srl->l3_bias);
  SENNA_free(srl->l4_weight);
  SENNA_free(srl->l4_bias);
  SENNA_free(srl->viterbi_score_init);
  SENNA_free(srl->viterbi_score_trans);
  
  /* extra inputs */
  SENNA_free(srl->sentence_posw);
  SENNA_free(srl->sentence_posv);    
  
  /* states */
  SENNA_free(srl->input_state);
  SENNA_free(srl->input_state_wcc);
  SENNA_free(srl->input_state_pv);
  SENNA_free(srl->input_state_pw);
  SENNA_free(srl->hidden_state1);
  SENNA_free(srl->hidden_state1_wcc);
  SENNA_free(srl->hidden_state1_pv);
  SENNA_free(srl->hidden_state1_pw);
  SENNA_free(srl->hidden_state2);
  SENNA_free(srl->hidden_state3);
  SENNA_free(srl->output_state);
  for(i = 0; i < srl->labels_size; i++)
    SENNA_free(srl->labels[i]);  
  SENNA_free(srl->labels);

  /* the end */
  SENNA_free(srl);
}
