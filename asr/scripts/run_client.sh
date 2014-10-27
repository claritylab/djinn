#/bin/bash

source ./cmd.sh

source ./path.sh

hostname=$1
portno=$2

njobs=1 # Number of parallel jobs
data_fmllr=exp/data-fmllr-tri3b # Root directory of fmllr features
data_dir=$data_fmllr/one_utt_test/ # Target directory to be decoded
gmm_dir=exp/tri3b/
gmm_ali_dir=exp/tri3b_ali/
dnn_dir=exp/dnn5b_pretrain-dbn_dnn/
target_dir=$dnn_dir/decode_one_utt_test/

# Running decode
steps/nnet/decode.sh --hostname=$hostname --portno=$portno \
                    --nj $njobs --cmd "$decode_cmd" --config conf/decode_dnn.config \
                    --nnet $dnn_dir/final.nnet --acwt 0.1 \
                    $gmm_dir/graph/ $data_dir $target_dir

