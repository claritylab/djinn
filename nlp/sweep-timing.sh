#!/bin/bash

pwd=$PWD

task=$1
batch=$2
reps=5

export PROF_REQ_TYPE=$task
export CUDA_VISIBLE_DEVICES=4
export BATCH=$batch

pkill dnn-server

# Prepare the batch input
./prepare-batch.sh $batch

for i in `seq 1 "$reps"`;
do
  export NUM_TRIAL=$i
  # Open up the server
  cd ../dnn/
  nvprof --print-gpu-trace --csv --devices 0 --log-file ./gpu_traces/trace.%q{PROF_REQ_TYPE}.%q{BATCH}.%q{NUM_TRIAL}.csv \
    ./dnn-server --portno 8398 --debug 0 --gpu 1 --csv ./trash.csv --threadcnt 2 &

  sleep 10

  cd $pwd
  ./run-nlp.sh $task 8398 input/batch-input.txt
  
done
