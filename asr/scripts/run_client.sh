#/bin/bash

# Script to run kaldi/asr client
# Yiping Kang (ypkang@umich.edu)
# 2014

source ./cmd.sh

source ./path.sh

hostname=$1 # ipaddress of the DNN server
portno=$2  # portno of the DNN server
service=$3 # true|false, to use DNN server or not

numquery=1 # The number of identical queries to sent continiously, default to 1

test_set=one_utt_test
njobs=1 # Number of parallel jobs
data_fmllr=exp/data-fmllr-tri3b # Root directory of fmllr features
data_dir=$data_fmllr/$test_set # Target directory to be decoded
gmm_dir=exp/tri3b/
dnn_dir=exp/dnn5b_pretrain-dbn_dnn/
target_dir=$dnn_dir/decode_${test_set}

# Make fMLLR features
steps/nnet/make_fmllr_feats.sh --nj $njobs --cmd "$train_cmd" \
        --transform-dir $gmm_dir/decode \
        $data_dir data/$test_set $gmm_dir $data_dir/log $data_dir/data || exit 1

# Running decode
steps/nnet/decode.sh --use-service $service --hostname $hostname --portno $portno --numquery $numquery \
                    --nj $njobs --cmd "$decode_cmd" --config conf/decode_dnn.config \
                    --nnet $dnn_dir/final.nnet --acwt 0.1 \
                    $gmm_dir/graph/ $data_dir $target_dir || exit 1
