#!/bin/bash

# Stop the MPS control daemon for each GPU and clean up /tmp

NGPUS=8

for ((i=0; i< $NGPUS; i++))
do
  pkill dnn-server
  echo $i
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
  echo "quit" | nvidia-cuda-mps-control
  rm -fr /tmp/mps_$i
  rm -fr /tmp/mps_log_$i
  pkill nvidia-cuda-mps-*
done
