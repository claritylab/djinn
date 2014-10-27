#include "SENNA_VBS.h"
#include "SENNA_utils.h"

SENNA_VBS* SENNA_VBS_new(const char *path)
{
  SENNA_VBS *vbs = SENNA_malloc(sizeof(SENNA_VBS), 1);
  FILE *f;
  float dummy;

  f = SENNA_fopen(path, "rb");

  SENNA_fread(&vbs->window_size, sizeof(int), 1, f);
  SENNA_fread_tensor_2d(&vbs->ll_word_weight, &vbs->ll_word_size, &vbs->ll_word_max_idx, f);
  SENNA_fread_tensor_2d(&vbs->ll_caps_weight, &vbs->ll_caps_size, &vbs->ll_caps_max_idx, f);
  SENNA_fread_tensor_2d(&vbs->ll_posl_weight, &vbs->ll_posl_size, &vbs->ll_posl_max_idx, f);
  SENNA_fread_tensor_2d(&vbs->l1_weight, &vbs->input_state_size, &vbs->hidden_state_size, f);
  SENNA_fread_tensor_1d(&vbs->l1_bias, &vbs->hidden_state_size, f);
  SENNA_fread_tensor_2d(&vbs->l2_weight, &vbs->hidden_state_size, &vbs->output_state_size, f);
  SENNA_fread_tensor_1d(&vbs->l2_bias, &vbs->output_state_size, f);

  SENNA_fread(&vbs->ll_word_padding_idx, sizeof(int), 1, f);
  SENNA_fread(&vbs->ll_caps_padding_idx, sizeof(int), 1, f);
  SENNA_fread(&vbs->ll_posl_padding_idx, sizeof(int), 1, f);

  SENNA_fread(&dummy, sizeof(float), 1, f);
  SENNA_fclose(f);

  if((int)dummy != 777)
    SENNA_error("vbs: data corrupted (or not IEEE floating computer)");

  vbs->input_state = NULL;
  vbs->hidden_state = SENNA_malloc(sizeof(float), vbs->hidden_state_size);
  vbs->output_state = NULL;
  vbs->labels = NULL;

  /* some info if you want verbose */
  SENNA_message("vbs: window size: %d", vbs->window_size);
  SENNA_message("vbs: vector size in word lookup table: %d", vbs->ll_word_size);
  SENNA_message("vbs: word lookup table size: %d", vbs->ll_word_max_idx);
  SENNA_message("vbs: vector size in caps lookup table: %d", vbs->ll_caps_size);
  SENNA_message("vbs: caps lookup table size: %d", vbs->ll_caps_max_idx);
  SENNA_message("vbs: vector size in pos lookup table: %d", vbs->ll_posl_size);
  SENNA_message("vbs: pos lookup table size: %d", vbs->ll_posl_max_idx);
  SENNA_message("vbs: number of hidden units: %d", vbs->hidden_state_size);
  SENNA_message("vbs: number of classes: %d", vbs->output_state_size);

  return vbs;
}

void SENNA_VBS_free(SENNA_VBS *vbs)
{
  SENNA_free(vbs->ll_word_weight);
  SENNA_free(vbs->ll_caps_weight);
  SENNA_free(vbs->ll_posl_weight);
  SENNA_free(vbs->l1_weight);
  SENNA_free(vbs->l1_bias);  
  SENNA_free(vbs->l2_weight);
  SENNA_free(vbs->l2_bias);
  
  SENNA_free(vbs->input_state);
  SENNA_free(vbs->hidden_state);
  SENNA_free(vbs->output_state);
  SENNA_free(vbs->labels);

  SENNA_free(vbs);
}
