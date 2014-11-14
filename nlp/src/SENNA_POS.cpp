#include <sys/time.h>
#include <unistd.h>
#include "SENNA_POS.h"
#include "SENNA_utils.h"
#include "SENNA_nn.h"
#include "socket.h"
#include <assert.h>

int* SENNA_POS_forward(SENNA_POS *pos, const int *sentence_words, const int *sentence_caps, const int *sentence_suff, int sentence_size, int socketfd)
{
  int idx;
  struct timeval tv1, tv2;

  gettimeofday(&tv1,NULL);
  pos->input_state = SENNA_realloc(pos->input_state,
                                   sizeof(float),
                                   (sentence_size+pos->window_size-1)*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)
                                   );
  pos->output_state = SENNA_realloc(pos->output_state,
                                    sizeof(float),
                                    sentence_size*pos->output_state_size
                                    );
  
  SENNA_nn_lookup(pos->input_state,
                  pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size,
                  pos->ll_word_weight,
                  pos->ll_word_size,
                  pos->ll_word_max_idx,
                  sentence_words,
                  sentence_size,
                  pos->ll_word_padding_idx,
                  (pos->window_size-1)/2
                  );
  SENNA_nn_lookup(pos->input_state+pos->ll_word_size,
                  pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size,
                  pos->ll_caps_weight,
                  pos->ll_caps_size,
                  pos->ll_caps_max_idx,
                  sentence_caps,
                  sentence_size,
                  pos->ll_caps_padding_idx,
                  (pos->window_size-1)/2
                  );
  SENNA_nn_lookup(pos->input_state+pos->ll_word_size+pos->ll_caps_size, 
                  pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size,
                  pos->ll_suff_weight,
                  pos->ll_suff_size,
                  pos->ll_suff_max_idx,
                  sentence_suff,
                  sentence_size,
                  pos->ll_suff_padding_idx,
                  (pos->window_size-1)/2
                  );
  gettimeofday(&tv2,NULL);
  pos->apptime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);

  gettimeofday(&tv1,NULL);

  char* input_data = (char*) malloc(sentence_size*(pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size))*sizeof(float));

  for(idx = 0; idx < sentence_size; idx++){
    memcpy((char*)(input_data+idx*(pos->window_size)*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)*sizeof(float)),
                    (char*)(pos->input_state+idx*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)),
                    pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)*sizeof(float));               
  }

  if(pos->service) {
   SOCKET_send(socketfd,
              input_data,
              sentence_size*(pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size))*sizeof(float),
              pos->debug
              );

   int rcvd = SOCKET_receive(socketfd,
              (char*)(pos->output_state),
              sentence_size*(pos->output_state_size)*sizeof(float),
              pos->debug
              );
  }

/*
  for(idx = 0; idx < sentence_size; idx++)
  {
      if(pos->service) {
        SOCKET_send(socketfd,
                    (char*)(pos->input_state+idx*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)),
                    sentence_size*(pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size))*sizeof(float),
                    pos->debug
                    );
        SOCKET_receive(socketfd,
                       (char*)(pos->output_state+idx*pos->output_state_size),
                       pos->output_state_size*sizeof(float),
                       pos->debug
                       );
      }
      else {
          SENNA_nn_linear(pos->hidden_state,
                  pos->hidden_state_size,
                  pos->l1_weight,
                  pos->l1_bias,
                  pos->input_state+idx*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size),
                  pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)
                  );
          SENNA_nn_hardtanh(pos->hidden_state,
                  pos->hidden_state,
                  pos->hidden_state_size
                  );
          SENNA_nn_linear(pos->output_state+idx*pos->output_state_size,
                  pos->output_state_size,
                  pos->l2_weight,
                  pos->l2_bias,
                  pos->hidden_state,
                  pos->hidden_state_size
                  );
      }
      pos->calls++;
  }
  */
  gettimeofday(&tv2,NULL);
  pos->dnntime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);

  gettimeofday(&tv1,NULL);
  pos->labels = SENNA_realloc(pos->labels, sizeof(int), sentence_size);
  SENNA_nn_viterbi(pos->labels,
                   pos->viterbi_score_init,
                   pos->viterbi_score_trans,
                   pos->output_state,
                   pos->output_state_size,
                   sentence_size
                   );
  gettimeofday(&tv2,NULL);
  pos->apptime += (tv2.tv_sec-tv1.tv_sec)*1000000 + (tv2.tv_usec-tv1.tv_usec);

  return pos->labels;
}

SENNA_POS* SENNA_POS_new(const char *path, const char *subpath)
{
  SENNA_POS *pos = SENNA_malloc(sizeof(SENNA_POS), 1);
  FILE *f;
  float dummy;

  memset(pos, 0, sizeof(SENNA_POS));

  f = SENNA_fopen(path, subpath, "rb");

  SENNA_fread(&pos->window_size, sizeof(int), 1, f);
  SENNA_fread_tensor_2d(&pos->ll_word_weight, &pos->ll_word_size, &pos->ll_word_max_idx, f);
  SENNA_fread_tensor_2d(&pos->ll_caps_weight, &pos->ll_caps_size, &pos->ll_caps_max_idx, f);
  SENNA_fread_tensor_2d(&pos->ll_suff_weight, &pos->ll_suff_size, &pos->ll_suff_max_idx, f);
  SENNA_fread_tensor_2d(&pos->l1_weight, &pos->input_state_size, &pos->hidden_state_size, f);
  SENNA_fread_tensor_1d(&pos->l1_bias, &pos->hidden_state_size, f);
  SENNA_fread_tensor_2d(&pos->l2_weight, &pos->hidden_state_size, &pos->output_state_size, f);
  SENNA_fread_tensor_1d(&pos->l2_bias, &pos->output_state_size, f);
  SENNA_fread_tensor_1d(&pos->viterbi_score_init, &pos->output_state_size, f);
  SENNA_fread_tensor_2d(&pos->viterbi_score_trans, &pos->output_state_size, &pos->output_state_size, f);

  SENNA_fread(&pos->ll_word_padding_idx, sizeof(int), 1, f);
  SENNA_fread(&pos->ll_caps_padding_idx, sizeof(int), 1, f);
  SENNA_fread(&pos->ll_suff_padding_idx, sizeof(int), 1, f);

  SENNA_fread(&dummy, sizeof(float), 1, f);
  SENNA_fclose(f);

  if((int)dummy != 777)
    SENNA_error("pos: data corrupted (or not IEEE floating computer)");

  pos->input_state = NULL;
  pos->hidden_state = SENNA_malloc(sizeof(float), pos->hidden_state_size);
  pos->output_state = NULL;
  pos->labels = NULL;

  /* some info if you want verbose */
  SENNA_message("pos: window size: %d", pos->window_size);
  SENNA_message("pos: vector size in word lookup table: %d", pos->ll_word_size);
  SENNA_message("pos: word lookup table size: %d", pos->ll_word_max_idx);
  SENNA_message("pos: vector size in caps lookup table: %d", pos->ll_caps_size);
  SENNA_message("pos: caps lookup table size: %d", pos->ll_caps_max_idx);
  SENNA_message("pos: vector size in suffix lookup table: %d", pos->ll_suff_size);
  SENNA_message("pos: suffix lookup table size: %d", pos->ll_suff_max_idx);
  SENNA_message("pos: number of hidden units: %d", pos->hidden_state_size);
  SENNA_message("pos: number of classes: %d", pos->output_state_size);

  pos->service = false;
  pos->debug = false;
  pos->socketfd = -1;
  pos->calls = 0;
  pos->dnntime = 0;
  pos->apptime = 0;

  return pos;
}

void SENNA_POS_free(SENNA_POS *pos)
{
  SENNA_free(pos->ll_word_weight);
  SENNA_free(pos->ll_caps_weight);
  SENNA_free(pos->ll_suff_weight);
  SENNA_free(pos->l1_weight);
  SENNA_free(pos->l1_bias);  
  SENNA_free(pos->l2_weight);
  SENNA_free(pos->l2_bias);
  SENNA_free(pos->viterbi_score_init);
  SENNA_free(pos->viterbi_score_trans);
  
  SENNA_free(pos->input_state);
  SENNA_free(pos->hidden_state);
  SENNA_free(pos->output_state);
  SENNA_free(pos->labels);

  SENNA_free(pos);
}
