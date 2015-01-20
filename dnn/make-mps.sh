#! /bin/bash

id=$1
mkdir /tmp/mps_$id
mkdir /tmp/mps_log_$id
export CUDA_VISIBLE_DEVICES=$id
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$id
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$id
nvidia-cuda-mps-control -d
