#ifndef SENNA_NN_H
#define SENNA_NN_H

void SENNA_nn_lookup(double *dest, int dest_stride, const double *wordweights, int wordsize, int maxwordidx, const int *wordindices, int nword, int padidx, int npad);
void SENNA_nn_hardtanh(double *output, double *input, int size);
void SENNA_nn_linear(double *output, int output_size, double *weights, double *biases, double *input, int input_size);
void SENNA_nn_max(double *value_, int *idx_, double *input, int input_size);
void SENNA_nn_temporal_convolution(double *output, int output_frame_size, double *weights, double *biases, double *input, int input_frame_size, int n_frames, int k_w);
void SENNA_nn_temporal_max_convolution(double *output, double *bias, double *input, int input_frame_size, int n_frames, int k_w);
void SENNA_nn_temporal_max(double *output, double *input, int N, int T);
void SENNA_nn_distance(int *dest, int idx, int max_idx, int sentence_size, int padding_size);
void SENNA_nn_viterbi(int *path, double *init, double *transition, double *emission, int N, int T);

#endif
