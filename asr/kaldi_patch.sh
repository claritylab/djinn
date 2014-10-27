#!/bin/sh

# Yiping Kang (ypkang@umich.edu)
# 2014

# This script path the relative source files in kaldi
# and copy over new scripts

# Change this into your kaldi root directory
kaldi_root=/home/ypkang/kaldi-trunk

# Copy over new scripts to egs/voxforge
voxforge_root=${kaldi_root}/egs/voxforge/s5

cp ./scripts/run_client.sh ${voxforge_root}/
cp ./scripts/run_dnn.sh ${voxforge_root}/local/

# Patch files
patch ${kaldi_root}/src/nnetbin/nnet-forward.cc < ./patch/nnet-forward.patch
patch ${kaldi_root}/src/nnetbin/Makefile < ./patch/Makefile.patch
patch ${voxforge_root}/steps/nnet/decode.sh < ./patch/decode.patch

# Copy over socket library
cp ./libsocket.a ${kaldi_root}/src/nnetbin/
cp ../socket-lib/socket.h ${kaldi_root}/src/nnetbin/

# Follow the README on how to starting running the application
