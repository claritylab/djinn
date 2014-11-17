#!/bin/bash

metrics="achieved_occupancy,sm_efficiency,warp_execution_efficiency"

# batch_size=( 1 )
batch_size=( 1 16 32 128 256 512 1024 2048 )
pwd=$PWD;

gpuid=7
export PROF_REQ_TYPE=$task
export CUDA_VISIBLE_DEVICES=$gpuid
port=$(( 7999 + $gpuid*100 ))
stats=$pwd/stats/
mkdir -p $stats
 
for task in imc dig; do
    for batch in "${batch_size[@]}";
    do
        echo $task $batch
        cd ../dnn/
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv $stats/timing_${task}_${batch}.csv --threadcnt 1 &
        sleep 20

        cd $pwd; 

        export glog_logtostderr=1
        export glog_log_dir=`pwd`/log
        ./img-client --hostname localhost \
        --task $task \
        --portno $port \
        --input input/${task}-input.bin \
        --num $batch \
        --flandmark data/flandmark.dat \
        --haar data/haar.xml \
        --debug 0
        sleep 5

        pkill nvprof
        pkill dnn-server
        cd ../dnn/
        nvprof --devices 0 \
        --metrics $metrics \
        --csv \
        --log-file log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt 1 &
        sleep 20

        cd $pwd; 

        export glog_logtostderr=1
        export glog_log_dir=`pwd`/log
        ./img-client --hostname localhost \
        --task $task \
        --portno $port \
        --input input/${task}-input.bin \
        --num $batch \
        --flandmark data/flandmark.dat \
        --haar data/haar.xml \
        --debug 0
        mv ../dnn/log.csv $stats/prof_${task}_${batch}.csv
        sleep 5

        pkill nvprof
        pkill dnn-server
        cd ../dnn/
        nvprof --devices 0 \
        --print-gpu-summary \
        --csv \
        --log-file log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt 1 &
        sleep 20

        cd $pwd; 

        export glog_logtostderr=1
        export glog_log_dir=`pwd`/log
        ./img-client --hostname localhost \
        --task $task \
        --portno $port \
        --input input/${task}-input.bin \
        --num $batch \
        --flandmark data/flandmark.dat \
        --haar data/haar.xml \
        --debug 0
        sleep 5
        mv ../dnn/log.csv $stats/all_${task}_${batch}.csv

    done
done
