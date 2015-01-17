#!/bin/sh

# Yiping Kang (ypkang@umich.edu)
# 2014

# This script path the relative source files in kaldi
# and copy over new scripts

# ATTENTION. Change this into your kaldi root directory before running the script
kaldi_root=/home/ypkang/git/kaldi-trunk-clean/

echo "Specified kaldi root is at ${kaldi_root}"

echo "Source files and scripts will be overwritten n ${kaldi_root}"

read -p "Do you want to proceed? [Y/n]" -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
  echo "Aborted on user input"
  exit 1
fi

echo "Start the patching process..."
echo ""

# Copy over new scripts to egs/voxforge
voxforge_root=${kaldi_root}/egs/voxforge/s5

echo "Start copying stuff..."

cp ./scripts/run_client.sh ${voxforge_root}/
cp ./scripts/decode.sh ${voxforge_root}/steps/nnet/
cp ./src/Makefile ${kaldi_root}/src/nnetbin/
cp ./scripts/path.sh ${voxforge_root}/
cp ./scripts/cmd.sh ${voxforge_root}/

echo "Done copy over voxforge related scripts"

# Patch source file
patch ${kaldi_root}/src/nnetbin/nnet-forward.cc < ./patch/nnet-forward.patch

echo "Done patch kaldi source code"

# Copy over socket library
cp ../socket-lib/libsocket.a ${kaldi_root}/src/nnetbin/
cp ../socket-lib/socket.h ${kaldi_root}/src/nnetbin/

echo "Done copy over socket library"

# Now we start to copy over voxforge related data

# 1. Copy over the test set. Two are provided. 
# one_utt_test only has one utterance and it is for testing purpose. 
# deepblue_test has several tenth of utterance, so it's better for timing and experiments.
# Bigger test set will be added later after I test them out
mkdir ${voxforge_root}/data/

cp -r voxforge_data/one_utt_test/ ${voxforge_root}/data/
cp -r voxforge_data/deepblue_test/ ${voxforge_root}/data/

echo "Done copy over selected test set"

# 2. Copy over the raw mfcc features. This will be referenced by the two data set above.
mkdir ${voxforge_root}/data_set/
cp -r voxforge_data/mfcc/ ${voxforge_root}/data_set/

echo "Done copy over raw mfcc features"

# 3. Copy over the graph/gmm dir 
mkdir ${voxforge_root}/exp/
cp -r voxforge_data/tri3b/ ${voxforge_root}/exp/

echo "Done copy over graph/gmm dir"

# 4. Copy over the directory contains feature transformation network and local dnn
cp -r voxforge_data/dnn5b_pretrain-dbn_dnn/ ${voxforge_root}/exp/

echo "Done copy over local model"

# 5. Finally, copy over the configuration file for decoding
cp voxforge_data/decode_dnn.config ${voxforge_root}/conf/

echo "Done copy over decode_dnn.config"

# Change the directory pointer in the feature scp files
SCP_FILES=${voxforge_root}/data/*/*.scp
for file in $SCP_FILES
do
  sed -i "s~/home/ypkang/kaldi-trunk~${kaldi_root}~g" $file
done

SCP_FILES=${voxforge_root}/data_set/mfcc/*.scp
for file in $SCP_FILES
do
  sed -i "s~/home/ypkang/kaldi-trunk~${kaldi_root}~g" $file
done

# Also change the data set pointer in path.sh
sed -i "s~/home/ypkang/kaldi-trunk/egs/voxforge/s5/data_set~${voxforge_root}/data_set~g" ${voxforge_root}/path.sh

echo "Done changing the directory pointer in scp files"

echo "You should now naviagte to src/nnetbin and re-make"
echo "Then you will be able to run the client in voxforge/s5/ with run_client.sh and arguments"
