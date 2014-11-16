#!/bin/bash

ifgpu=$1


req_type=$2

export PROF_REQ_TYPE=$req_type
export CUDA_VISIBLE_DEVICES=1

#nvprof --output-profile ./prof.%q{PROF_REQ_TYPE}.%p.out --devices 0 --log-file ./profiling/prof.%q{PROF_REQ_TYPE}.%p.out ./dnn-server --portno 8398 --debug 1 --gpu $1
# ./dnn-server --portno 8398 --debug 1 --gpu true

if [ "$ifgpu" -eq 1 ] ;
then
        nvprof --print-gpu-trace --csv --devices 0 --log-file ./prof.%q{PROF_REQ_TYPE}.%p.out \
          ./dnn-server --portno 8398 --debug 1 --gpu $1 --csv ./timing.csv
        echo ""
        echo "GPU Trace"
        #cat ./prof.$req_type*.out
        #cat ./prof.$req_type.*.out | grep "memcpy" | awk ' {print $1 " " $2 " " $15 " " $16 " " $17} '

        #mv ./prof.$req_type.*.out ./profiling/
else
        ./dnn-server --portno 8398 --debug 1 --gpu 0 --csv ./timing.csv
fi

