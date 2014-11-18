#!/bin/bash

metrics="achieved_occupancy,sm_efficiency,ipc,stall_memory_throttle,dram_utilization"

# batch_size=( 1 )
batch_size=( 1 16 32 128 256 512 1024 )
pwd=$PWD;

gpuid=7
export PROF_REQ_TYPE=$task
export CUDA_VISIBLE_DEVICES=$gpuid
port=$(( 7999 + $gpuid*100 ))
stats=$pwd/stats/
mkdir -p $stats
# rm -rf $stats/*
 
for task in face; do
    for batch in "${batch_size[@]}";
    do
        echo $task $batch
        cd ../dnn/
        if [ "$task" == "face" ]; then
            ./change_batch.sh face $batch
        fi
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv $stats/timing_${task}_${batch}.csv --threadcnt 1 &
        sid=$!
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

        kill $sid
        cd ../dnn/
        nvprof --devices 0 \
        --metrics $metrics \
        --csv \
        --log-file log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt 1 &
        sid=$!
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

        kill $sid
        cd ../dnn/
        nvprof --devices 0 \
        --print-gpu-summary \
        --csv \
        --log-file log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt 1 &
        sid=$!
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
        mv ../dnn/log.csv $stats/all_${task}_${batch}.csv
        kill $sid
    done
done
