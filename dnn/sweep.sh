#!/bin/bash

for model in imc dig dig-10 VGG face asr pos ner chk;do
#for model in imc face;do
  for run in {1..9};do
    ./local_run.sh $model true false micro-timing/timing.csv 1
    sleep 2
    ./layer_run.sh $model true false micro-timing/layer_timing.csv 1 micro-timing/gpu_${model}_layer.csv
    sleep 2
#    ./local_run.sh $model false false micro-timing/timing.csv 1
#    sleep 2
#    ./layer_run.sh $model false false micro-timing/layer_timing.csv 1 micro-timing/cpu_${model}_layer.csv
#    sleep 2
  done
done
