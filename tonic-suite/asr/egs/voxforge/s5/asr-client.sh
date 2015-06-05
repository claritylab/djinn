#/bin/bash

source ./cmd.sh

source ./path.sh

root_dataset=data_set
data_fmllr=exp/data-fmllr-tri3b # Root directory of fmllr features
data_dir=$data_fmllr/local # Target directory to be decoded
graph_dir=exp/tri3b/
dnn_dir=exp/djinn_decode/
target_dir=$dnn_dir/decode_local

# arguments to pass in
djinn=true # Use DjiNN service ?
hostname=localhost # server hostname
portno=8080 # server port
common=../../../../../common/
network=asr.prototxt
weights=asr.caffemodel
input=asr-list.txt

# Prepare data
local/voxforge_data_prep.sh ${root_dataset}/selected

# Make mfcc features
mfccdir=data/mfcc

steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 \
  data/local exp/make_mfcc/local $mfccdir || exit 1;

steps/compute_cmvn_stats.sh data/local exp/make_mfcc/local $mfccdir || exit 1;

# Make fMLLR features
steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
        --transform-dir $graph_dir/decode \
        $data_dir data/local $graph_dir $data_dir/log $data_dir/data || exit 1;

# Running decode
steps/nnet/decode.sh --djinn $djinn --hostname $hostname --portno $portno \
                    --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config \
                    --nnet $dnn_dir/final.nnet --acwt 0.1 \
                    --common $common --network $network --weights $weights \
                    $graph_dir/graph/ $data_dir $target_dir || exit 1;
